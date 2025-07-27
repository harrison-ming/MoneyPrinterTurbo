import math
import os.path
import re
import requests
import subprocess
from os import path

from loguru import logger
from moviepy import (
    VideoFileClip,
    concatenate_videoclips,
)

from app.config import config
from app.models import const
from app.models.schema import VideoConcatMode, VideoParams, MaterialInfo
from app.services import llm, material, subtitle, video, voice
from app.services import state as sm
from app.utils import utils


def generate_script(task_id, params):
    logger.info("\n\n## generating video script")
    video_script = params.video_script.strip()
    if not video_script:
        video_script = llm.generate_script(
            video_subject=params.video_subject,
            language=params.video_language,
            paragraph_number=params.paragraph_number,
        )
    else:
        logger.debug(f"video script: \n{video_script}")

    if not video_script:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error("failed to generate video script.")
        return None

    return video_script


def generate_terms(task_id, params, video_script):
    logger.info("\n\n## generating video terms")
    video_terms = params.video_terms
    if not video_terms:
        video_terms = llm.generate_terms(
            video_subject=params.video_subject, video_script=video_script, amount=5
        )
    else:
        if isinstance(video_terms, str):
            video_terms = [term.strip() for term in re.split(r"[,，]", video_terms)]
        elif isinstance(video_terms, list):
            video_terms = [term.strip() for term in video_terms]
        else:
            raise ValueError("video_terms must be a string or a list of strings.")

        logger.debug(f"video terms: {utils.to_json(video_terms)}")

    if not video_terms:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error("failed to generate video terms.")
        return None

    return video_terms


def save_script_data(task_id, video_script, video_terms, params):
    script_file = path.join(utils.task_dir(task_id), "script.json")
    script_data = {
        "script": video_script,
        "search_terms": video_terms,
        "params": params,
    }

    with open(script_file, "w", encoding="utf-8") as f:
        f.write(utils.to_json(script_data))


def generate_audio(task_id, params, video_script, index):
    logger.info("\n\n## generating audio")
    audio_file = path.join(utils.task_dir(task_id), f"audio-{index}.mp3")
    logger.info(
        f"audio file: {audio_file}, voice name: {params.voice_name}, voice rate: {params.voice_rate}, video script: {video_script}"
    )
    sub_maker = voice.tts(
        text=video_script,
        voice_name=voice.parse_voice_name(params.voice_name),
        voice_rate=params.voice_rate,
        voice_file=audio_file,
    )
    if sub_maker is None:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error(
            """failed to generate audio:
1. check if the language of the voice matches the language of the video script.
2. check if the network is available. If you are in China, it is recommended to use a VPN and enable the global traffic mode.
        """.strip()
        )
        return None, None, None

    audio_duration = math.ceil(voice.get_audio_duration(sub_maker))
    return audio_file, audio_duration, sub_maker


def generate_subtitle(task_id, params, video_script, sub_maker, audio_file, index):
    if not params.subtitle_enabled:
        return ""

    subtitle_path = path.join(utils.task_dir(task_id), f"subtitle-{index}.srt")
    subtitle_provider = config.app.get("subtitle_provider", "edge").strip().lower()
    logger.info(f"\n\n## generating subtitle, provider: {subtitle_provider}")

    subtitle_fallback = False
    if subtitle_provider == "edge":
        voice.create_subtitle(
            text=video_script, sub_maker=sub_maker, subtitle_file=subtitle_path
        )
        if not os.path.exists(subtitle_path):
            subtitle_fallback = True
            logger.warning("subtitle file not found, fallback to whisper")

    if subtitle_provider == "whisper" or subtitle_fallback:
        subtitle.create(audio_file=audio_file, subtitle_file=subtitle_path)
        logger.info("\n\n## correcting subtitle")
        subtitle.correct(subtitle_file=subtitle_path, video_script=video_script)

    subtitle_lines = subtitle.file_to_subtitles(subtitle_path)
    if not subtitle_lines:
        logger.warning(f"subtitle file is invalid: {subtitle_path}")
        return ""

    return subtitle_path


def generate_images(task_id: str, params: VideoParams):
    story_list = llm.generate_story_with_images(story=params.video_script.strip(), language=params.language, 
                                                    segments=params.segments, resolution=params.resolution)
    materials = []
    for i, scene in enumerate(story_list, 1):
        if scene.get("url"):     
            image_path = path.join(utils.task_dir(task_id), f"{i}.png")
            try:
                response = requests.get(scene["url"])
                if response.status_code == 200:
                    with open(image_path, "wb") as f:
                        f.write(response.content)
                    logger.info(f"Downloaded image {i} to {image_path}")
                    item = MaterialInfo(
                        url=image_path,
                    )
                    materials.append(item)
            except Exception as e:
                logger.error(f"Failed to download image {i}: {e}")
    
    return materials


def get_video_materials(task_id, params, video_terms, audio_duration, index):
    script_segments = params.script_segments
    if script_segments:
        materials = []
        item = MaterialInfo(url=script_segments[index - 1]["url"])
        materials.append(item)
        
        params.video_materials = materials
        logger.info("\n\n## preprocess materials passed from script")
        materials = video.preprocess_video(
            materials=params.video_materials, 
            clip_duration=params.video_clip_duration,
            image_to_video=script_segments[index - 1].get("image_to_video", True)
        )
        if not materials:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                "no valid materials found, please check the materials and try again."
            )
            return None
        return [material_info.url for material_info in materials]
    elif params.video_source == "ai":
        logger.info("\n\n## generating images for story")
        materials = generate_images(task_id, params)
        if not materials:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                "no valid images generated by AI."
            )
            return None
        
        params.video_materials = materials
        logger.info("\n\n## preprocess AI materials")
        materials = video.preprocess_video(
            materials=params.video_materials, clip_duration=params.video_clip_duration
        )
        if not materials:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                "no valid materials generated for AI images."
            )
            return None
        
        return [material_info.url for material_info in materials]
    elif params.video_source == "local":
        logger.info("\n\n## preprocess local materials")
        materials = video.preprocess_video(
            materials=params.video_materials, clip_duration=params.video_clip_duration
        )
        if not materials:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                "no valid materials found, please check the materials and try again."
            )
            return None
        return [material_info.url for material_info in materials]
    else:
        logger.info(f"\n\n## downloading videos from {params.video_source}")
        downloaded_videos = material.download_videos(
            task_id=task_id,
            search_terms=video_terms,
            source=params.video_source,
            video_aspect=params.video_aspect,
            video_contact_mode=params.video_concat_mode,
            audio_duration=audio_duration * params.video_count,
            max_clip_duration=params.video_clip_duration,
        )
        if not downloaded_videos:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                "failed to download videos, maybe the network is not available. if you are in China, please use a VPN."
            )
            return None
        return downloaded_videos


def generate_final_videos(
    task_id, params, downloaded_videos, audio_file, subtitle_path, nth
):
    final_video_paths = []
    combined_video_paths = []
    video_concat_mode = (
        params.video_concat_mode if params.video_count == 1 else VideoConcatMode.random
    )
    video_transition_mode = params.video_transition_mode

    _progress = 50
    
    # Check if we're in script_segments mode
    script_segments = params.script_segments if params else None
    if script_segments:
        # In script_segments mode, generate only one video per segment (no video_count loop)
        logger.info(f"script_segments mode: generating single video for segment {nth}")
        combined_video_path = path.join(
            utils.task_dir(task_id), f"combined-1-{nth}.mp4"
        )
        logger.info(f"\n\n## combining video for segment {nth} => {combined_video_path}")
        video.combine_videos(
            combined_video_path=combined_video_path,
            video_paths=downloaded_videos,
            audio_file=audio_file,
            video_aspect=params.video_aspect,
            video_concat_mode=video_concat_mode,
            video_transition_mode=video_transition_mode,
            max_clip_duration=params.video_clip_duration,
            threads=params.n_threads,
            params=params,
            nth=nth,
        )
        
        _progress += 25
        sm.state.update_task(task_id, progress=_progress)

        final_video_path = path.join(utils.task_dir(task_id), f"final-1-{nth}.mp4")

        logger.info(f"\n\n## generating video for segment {nth} => {final_video_path}")
        video.generate_video(
            video_path=combined_video_path,
            audio_path=audio_file,
            subtitle_path=subtitle_path,
            output_file=final_video_path,
            params=params,
        )

        _progress += 25
        sm.state.update_task(task_id, progress=_progress)

        final_video_paths.append(final_video_path)
        combined_video_paths.append(combined_video_path)
    else:
        # Original logic: generate multiple video variants
        for i in range(params.video_count):
            index = i + 1
            combined_video_path = path.join(
                utils.task_dir(task_id), f"combined-{index}-{nth}.mp4"
            )
            logger.info(f"\n\n## combining video: {index} => {combined_video_path}")
            video.combine_videos(
                combined_video_path=combined_video_path,
                video_paths=downloaded_videos,
                audio_file=audio_file,
                video_aspect=params.video_aspect,
                video_concat_mode=video_concat_mode,
                video_transition_mode=video_transition_mode,
                max_clip_duration=params.video_clip_duration,
                threads=params.n_threads,
                params=params,
                nth=nth,
            )

            _progress += 50 / params.video_count / 2
            sm.state.update_task(task_id, progress=_progress)

            final_video_path = path.join(utils.task_dir(task_id), f"final-{index}-{nth}.mp4")

            logger.info(f"\n\n## generating video: {index} => {final_video_path}")
            video.generate_video(
                video_path=combined_video_path,
                audio_path=audio_file,
                subtitle_path=subtitle_path,
                output_file=final_video_path,
                params=params,
            )

            _progress += 50 / params.video_count / 2
            sm.state.update_task(task_id, progress=_progress)

            final_video_paths.append(final_video_path)
            combined_video_paths.append(combined_video_path)

    return final_video_paths, combined_video_paths


def start(task_id, params: VideoParams, stop_at: str = "video"):
    logger.info(f"start task: {task_id}, stop_at: {stop_at}")
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=5)

    if type(params.video_concat_mode) is str:
        params.video_concat_mode = VideoConcatMode(params.video_concat_mode)

    final_video_paths = []
    combined_video_paths = []
    audio_files = []
    audio_durations = []
    subtitle_paths = []
    downloaded_videos = []
    video_scripts = []
    
    video_terms = ""

    # 1. Generate script
    script_segments = params.script_segments
    if not script_segments:
        video_script = generate_script(task_id, params)
        if not video_script or "Error: " in video_script:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            return

        sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=10)

        if stop_at == "script":
            sm.state.update_task(
                task_id, state=const.TASK_STATE_COMPLETE, progress=100, script=video_script
            )
            return {"script": video_script}

        # 2. Generate terms
        if params.video_source != "local":
            video_terms = generate_terms(task_id, params, video_script)
            if not video_terms:
                sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
                return

        save_script_data(task_id, video_script, video_terms, params)

        if stop_at == "terms":
            sm.state.update_task(
                task_id, state=const.TASK_STATE_COMPLETE, progress=100, terms=video_terms
            )
            return {"script": video_script, "terms": video_terms}

        sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=20)

        sub_final_video_paths, sub_combined_video_paths, sub_audio_file, sub_audio_duration, sub_subtitle_path, sub_downloaded_videos = step_3_to_step_6(
            task_id, params, video_script, None, 0, video_terms, stop_at, index=1
        )
        final_video_paths.extend(sub_final_video_paths)
        combined_video_paths.extend(sub_combined_video_paths)
        audio_files.append(sub_audio_file)
        audio_durations.append(sub_audio_duration)
        subtitle_paths.append(sub_subtitle_path)
        downloaded_videos.extend(sub_downloaded_videos)
        video_scripts.append(video_script)
    else:
        logger.info(f"script segments exists, skipping script and term generation")
        for index, segment in enumerate(script_segments, 1):
            video_script = segment["script"]
            pre_generated_audio_file = segment.get("audio_file", None)
            pre_generated_audio_duration = segment.get("audio_duration", None)
            logger.info(f"script segment: {index} => {video_script}")
            sub_final_video_paths, sub_combined_video_paths, sub_audio_file, sub_audio_duration, sub_subtitle_path, sub_downloaded_videos = step_3_to_step_6(
                task_id, params, video_script, pre_generated_audio_file, pre_generated_audio_duration, "", stop_at, index
            )

            final_video_paths.extend(sub_final_video_paths)
            combined_video_paths.extend(sub_combined_video_paths)
            audio_files.append(sub_audio_file)
            audio_durations.append(sub_audio_duration)
            subtitle_paths.append(sub_subtitle_path)
            downloaded_videos.extend(sub_downloaded_videos)
            video_scripts.append(video_script)

        # logger.info("combining final videos into one")
        # clips = [VideoFileClip(path) for path in final_video_paths]

        # # Concatenate the clips
        # final_clip = concatenate_videoclips(clips)

        # # Export the final video
        # combined_final_video_path = path.join(
        #     utils.task_dir(task_id), f"combined_final_video.mp4"
        # )
        # final_clip.write_videofile(combined_final_video_path, codec="libx264", audio_codec="aac")
        # final_video_paths = [combined_final_video_path]
        

        logger.info("combining final videos into one")

        # Create concat list file
        concat_list_path = os.path.join(utils.task_dir(task_id), "concat_final.txt")
        with open(concat_list_path, "w") as f:
            for path in final_video_paths:
                f.write(f"file '{os.path.abspath(path)}'\n")

        # Final output path
        combined_final_video_path = os.path.join(utils.task_dir(task_id), "combined_final_video.mp4")

        # FFmpeg concat without re-encoding
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            combined_final_video_path
        ])

        # Clean up
        os.remove(concat_list_path)

        # Update final paths list
        final_video_paths = [combined_final_video_path]
        logger.info(f"Final combined video saved to: {combined_final_video_path}")


    # # 3. Generate audio
    # audio_file, audio_duration, sub_maker = generate_audio(
    #     task_id, params, video_script
    # )
    # if not audio_file:
    #     sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
    #     return

    # sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=30)

    # if stop_at == "audio":
    #     sm.state.update_task(
    #         task_id,
    #         state=const.TASK_STATE_COMPLETE,
    #         progress=100,
    #         audio_file=audio_file,
    #     )
    #     return {"audio_file": audio_file, "audio_duration": audio_duration}

    # # 4. Generate subtitle
    # subtitle_path = generate_subtitle(
    #     task_id, params, video_script, sub_maker, audio_file
    # )

    # if stop_at == "subtitle":
    #     sm.state.update_task(
    #         task_id,
    #         state=const.TASK_STATE_COMPLETE,
    #         progress=100,
    #         subtitle_path=subtitle_path,
    #     )
    #     return {"subtitle_path": subtitle_path}

    # sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=40)

    # # 5. Get video materials
    # downloaded_videos = get_video_materials(
    #     task_id, params, video_terms, audio_duration
    # )
    # if not downloaded_videos:
    #     sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
    #     return

    # if stop_at == "materials":
    #     sm.state.update_task(
    #         task_id,
    #         state=const.TASK_STATE_COMPLETE,
    #         progress=100,
    #         materials=downloaded_videos,
    #     )
    #     return {"materials": downloaded_videos}

    # sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=50)

    # # 6. Generate final videos
    # final_video_paths, combined_video_paths = generate_final_videos(
    #     task_id, params, downloaded_videos, audio_file, subtitle_path
    # )

    # if not final_video_paths:
    #     sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
    #     return

    # logger.success(
    #     f"task {task_id} finished, generated {len(final_video_paths)} videos."
    # )

    kwargs = {
        "videos": final_video_paths,
        "combined_videos": combined_video_paths,
        "scripts": video_scripts,
        "terms": video_terms,
        "audio_files": audio_files,
        "audio_durations": audio_durations,
        "subtitle_paths": subtitle_paths,
        "materials": downloaded_videos,
    }
    sm.state.update_task(
        task_id, state=const.TASK_STATE_COMPLETE, progress=100, **kwargs
    )
    return kwargs


def step_3_to_step_6(task_id, params, video_script, pre_generated_audio_file, pre_generated_audio_duration, video_terms, stop_at, index):
    # 3. Generate audio
    audio_file, audio_duration, sub_maker = generate_audio(
        task_id, params, video_script, index
    )
    if not audio_file:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=30)

    if stop_at == "audio":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            audio_file=audio_file,
        )
        return {"audio_file": audio_file, "audio_duration": audio_duration}

    # 4. Generate subtitle
    subtitle_path = generate_subtitle(
        task_id, params, video_script, sub_maker, audio_file, index
    )

    if stop_at == "subtitle":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            subtitle_path=subtitle_path,
        )
        return {"subtitle_path": subtitle_path}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=40)

    if pre_generated_audio_file is not None and pre_generated_audio_duration is not None:
        logger.info(f"Using pre-generated audio file: {pre_generated_audio_file}")
        # Use pre-generated audio file and duration
        audio_file = pre_generated_audio_file
        audio_duration = pre_generated_audio_duration

    # 5. Get video materials
    downloaded_videos = get_video_materials(
        task_id, params, video_terms, audio_duration, index
    )
    if not downloaded_videos:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    if stop_at == "materials":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            materials=downloaded_videos,
        )
        return {"materials": downloaded_videos}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=50)

    # 6. Generate final videos
    final_video_paths, combined_video_paths = generate_final_videos(
        task_id, params, downloaded_videos, audio_file, subtitle_path, index
    )

    if not final_video_paths:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    logger.success(
        f"task {task_id} finished, generated {len(final_video_paths)} videos."
    )

    return final_video_paths, combined_video_paths, audio_file, audio_duration, subtitle_path, downloaded_videos


if __name__ == "__main__":
    task_id = "task_id"
    params = VideoParams(
        video_subject="金钱的作用",
        voice_name="zh-CN-XiaoyiNeural-Female",
        voice_rate=1.0,
    )
    start(task_id, params, stop_at="video")
