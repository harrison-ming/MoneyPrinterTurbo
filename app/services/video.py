import glob
import itertools
import os
import random
import gc
import shutil
import subprocess
from typing import List
from loguru import logger
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    afx,
    concatenate_videoclips,
)
from moviepy.video.tools.subtitles import SubtitlesClip
from PIL import ImageFont

from app.models import const
from app.models.schema import (
    MaterialInfo,
    VideoAspect,
    VideoConcatMode,
    VideoParams,
    VideoTransitionMode,
)
from app.services.utils import video_effects
from app.utils import utils


def _preprocess_image_for_imageio(image_path: str) -> str:
    """
    预处理图片以确保与imageio/FFMPEG兼容
    返回处理后的图片路径（可能是原路径或临时文件路径）
    """
    try:
        from PIL import Image
        import tempfile
        import os

        # 检查是否需要处理
        try:
            # 尝试直接读取，如果成功就不需要处理
            img = Image.open(image_path)
            if img.mode == 'RGB':
                # 如果已经是RGB模式，尝试直接使用
                return image_path
        except Exception:
            pass

        # 需要标准化处理
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 创建临时文件
        temp_dir = tempfile.gettempdir()
        temp_filename = f"imageio_compat_{os.path.basename(image_path)}.jpg"
        temp_path = os.path.join(temp_dir, temp_filename)

        # 保存标准化版本
        img.save(temp_path, 'JPEG', quality=95, optimize=True)

        logger.info(f"Image preprocessed for imageio compatibility: {image_path} -> {temp_path}")
        return temp_path

    except Exception as e:
        logger.warning(f"Failed to preprocess image, using original: {e}")
        return image_path


class SubClippedVideoClip:
    def __init__(self, file_path, start_time=None, end_time=None, width=None, height=None, duration=None):
        self.file_path = file_path
        self.start_time = start_time
        self.end_time = end_time
        self.width = width
        self.height = height
        if duration is None:
            self.duration = end_time - start_time
        else:
            self.duration = duration

    def __str__(self):
        return f"SubClippedVideoClip(file_path={self.file_path}, start_time={self.start_time}, end_time={self.end_time}, duration={self.duration}, width={self.width}, height={self.height})"


audio_codec = "aac"
video_codec = "libx264"
fps = 30


def close_clip(clip):
    if clip is None:
        return

    try:
        # close main resources
        if hasattr(clip, 'reader') and clip.reader is not None:
            clip.reader.close()

        # close audio resources
        if hasattr(clip, 'audio') and clip.audio is not None:
            if hasattr(clip.audio, 'reader') and clip.audio.reader is not None:
                clip.audio.reader.close()
            del clip.audio

        # close mask resources
        if hasattr(clip, 'mask') and clip.mask is not None:
            if hasattr(clip.mask, 'reader') and clip.mask.reader is not None:
                clip.mask.reader.close()
            del clip.mask

        # handle child clips in composite clips
        if hasattr(clip, 'clips') and clip.clips:
            for child_clip in clip.clips:
                if child_clip is not clip:  # avoid possible circular references
                    close_clip(child_clip)

        # clear clip list
        if hasattr(clip, 'clips'):
            clip.clips = []

    except Exception as e:
        logger.error(f"failed to close clip: {str(e)}")

    del clip
    gc.collect()


def delete_files(files: List[str] | str):
    if isinstance(files, str):
        files = [files]

    for file in files:
        try:
            os.remove(file)
        except:
            pass


def get_bgm_file(bgm_type: str = "random", bgm_file: str = ""):
    if not bgm_type:
        return ""

    if bgm_file and os.path.exists(bgm_file):
        return bgm_file

    if bgm_type == "random":
        suffix = "*.mp3"
        song_dir = utils.song_dir()
        files = glob.glob(os.path.join(song_dir, suffix))
        return random.choice(files)

    return ""


def combine_videos(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    video_concat_mode: VideoConcatMode = VideoConcatMode.random,
    video_transition_mode: VideoTransitionMode = None,
    max_clip_duration: int = 5,
    threads: int = 2,
    params: VideoParams = None,
    nth: int = 1,
) -> str:
    audio_clip = AudioFileClip(audio_file)
    audio_duration = audio_clip.duration
    logger.info(f"audio duration: {audio_duration} seconds")

    # Check for script_segments mode
    script_segments = params.script_segments if params else None
    if script_segments:
        logger.info("script_segments mode detected, using enhanced processing")
        return _handle_script_segments_mode(
            combined_video_path, video_paths, audio_file, audio_duration,
            video_aspect, video_concat_mode, video_transition_mode,
            max_clip_duration, threads, params, nth
        )

    # Original logic for backward compatibility
    # Required duration of each clip
    # req_dur = audio_duration / len(video_paths)
    # req_dur = max_clip_duration
    req_dur = min(audio_duration, max_clip_duration)
    logger.info(f"each clip will be maximum {req_dur} seconds long")
    output_dir = os.path.dirname(combined_video_path)

    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()

    processed_clips = []
    subclipped_items = []
    video_duration = 0
    for video_path in video_paths:
        clip = VideoFileClip(video_path)
        clip_duration = clip.duration
        clip_w, clip_h = clip.size
        close_clip(clip)

        start_time = 0
        selected_end_time = clip_duration
        script_segments = params.script_segments
        if audio_duration < clip_duration and script_segments:
            script_segment = script_segments[nth - 1]
            if script_segment:
                start_time = script_segment.get("start_time", 0)
                selected_end_time = min(script_segment.get("end_time", clip_duration), start_time + audio_duration)
                logger.info(f"using script segment: {script_segment}, start: {start_time}, end: {selected_end_time}")

        while start_time < selected_end_time:
            end_time = min(start_time + max_clip_duration, clip_duration)
            if clip_duration - start_time >= max_clip_duration:
                subclipped_items.append(SubClippedVideoClip(file_path= video_path, start_time=start_time, end_time=end_time, width=clip_w, height=clip_h))
            start_time = end_time
            # if video_concat_mode.value == VideoConcatMode.sequential.value:
                # break

    # random subclipped_items order
    if video_concat_mode.value == VideoConcatMode.random.value:
        random.shuffle(subclipped_items)

    logger.debug(f"total subclipped items: {len(subclipped_items)}")

    # Add downloaded clips over and over until the duration of the audio (max_duration) has been reached
    for i, subclipped_item in enumerate(subclipped_items):
        if video_duration >= audio_duration:
            break

        logger.debug(f"processing clip {i+1}: {subclipped_item.width}x{subclipped_item.height}, current duration: {video_duration:.2f}s, remaining: {audio_duration - video_duration:.2f}s")

        try:
            clip = VideoFileClip(subclipped_item.file_path).subclipped(subclipped_item.start_time, subclipped_item.end_time)
            clip_duration = clip.duration
            # Not all videos are same size, so we need to resize them
            clip_w, clip_h = clip.size
            if clip_w != video_width or clip_h != video_height:
                clip_ratio = clip.w / clip.h
                video_ratio = video_width / video_height
                logger.debug(f"resizing clip, source: {clip_w}x{clip_h}, ratio: {clip_ratio:.2f}, target: {video_width}x{video_height}, ratio: {video_ratio:.2f}")

                if clip_ratio == video_ratio:
                    clip = clip.resized(new_size=(video_width, video_height))
                else:
                    if clip_ratio > video_ratio:
                        scale_factor = video_width / clip_w
                    else:
                        scale_factor = video_height / clip_h

                    new_width = int(clip_w * scale_factor)
                    new_height = int(clip_h * scale_factor)

                    background = ColorClip(size=(video_width, video_height), color=(0, 0, 0)).with_duration(clip_duration)
                    clip_resized = clip.resized(new_size=(new_width, new_height)).with_position("center")
                    clip = CompositeVideoClip([background, clip_resized])

            shuffle_side = random.choice(["left", "right", "top", "bottom"])
            if video_transition_mode.value == VideoTransitionMode.none.value:
                clip = clip
            elif video_transition_mode.value == VideoTransitionMode.fade_in.value:
                clip = video_effects.fadein_transition(clip, 1)
            elif video_transition_mode.value == VideoTransitionMode.fade_out.value:
                clip = video_effects.fadeout_transition(clip, 1)
            elif video_transition_mode.value == VideoTransitionMode.slide_in.value:
                clip = video_effects.slidein_transition(clip, 1, shuffle_side)
            elif video_transition_mode.value == VideoTransitionMode.slide_out.value:
                clip = video_effects.slideout_transition(clip, 1, shuffle_side)
            elif video_transition_mode.value == VideoTransitionMode.shuffle.value:
                transition_funcs = [
                    lambda c: video_effects.fadein_transition(c, 1),
                    lambda c: video_effects.fadeout_transition(c, 1),
                    lambda c: video_effects.slidein_transition(c, 1, shuffle_side),
                    lambda c: video_effects.slideout_transition(c, 1, shuffle_side),
                ]
                shuffle_transition = random.choice(transition_funcs)
                clip = shuffle_transition(clip)

            if clip.duration > max_clip_duration:
                clip = clip.subclipped(0, max_clip_duration)

            # wirte clip to temp file
            clip_file = f"{output_dir}/temp-clip-{i+1}.mp4"
            logger.info(f"writing clip {i+1} to {clip_file}, duration: {clip.duration:.2f}s, size: {clip_w}x{clip_h}")
            clip.write_videofile(clip_file, logger=None, fps=fps, codec=video_codec)
            logger.info(f"clip {i+1} written to {clip_file}")

            close_clip(clip)

            processed_clips.append(SubClippedVideoClip(file_path=clip_file, duration=clip.duration, width=clip_w, height=clip_h))
            video_duration += clip.duration

        except Exception as e:
            logger.error(f"failed to process clip: {str(e)}")

    # loop processed clips until the video duration matches or exceeds the audio duration.
    if video_duration > audio_duration:
        logger.warning(f"video duration ({video_duration:.2f}s) exceeds audio duration ({audio_duration:.2f}s), trimming clips to match audio length.")
        # Trim the last clip to match the audio duration
        if processed_clips:
            last_clip = processed_clips[-1]
            last_clip_updated_duration = last_clip.duration - (video_duration - audio_duration)
            last_clip.start_time = 0
            last_clip.end_time = last_clip_updated_duration
            last_clip.duration = last_clip_updated_duration
            logger.info(f"trimmed last clip to {last_clip.duration:.2f}s")
    elif video_duration < audio_duration:
        logger.warning(f"video duration ({video_duration:.2f}s) is shorter than audio duration ({audio_duration:.2f}s), looping clips to match audio length.")
        base_clips = processed_clips.copy()
        for clip in itertools.cycle(base_clips):
            if video_duration >= audio_duration:
                break
            if video_duration + clip.duration >= audio_duration:
                processed_clips.append(
                    SubClippedVideoClip(
                        file_path=clip.file_path,
                        start_time=0,
                        end_time=audio_duration - video_duration,
                        width=clip.width,
                        height=clip.height,
                        duration=audio_duration - video_duration
                    )
                )
                video_duration = audio_duration
                logger.info(f"reached audio duration limit, breaking loop at {video_duration:.2f}s")
                break
            processed_clips.append(clip)
            video_duration += clip.duration
        logger.info(f"video duration: {video_duration:.2f}s, audio duration: {audio_duration:.2f}s, looped {len(processed_clips)-len(base_clips)} clips")
    else:
        logger.info(f"video duration ({video_duration:.2f}s) matches audio duration ({audio_duration:.2f}s), no looping required.")
    # merge video clips progressively, avoid loading all videos at once to avoid memory overflow
    logger.info("starting clip merging process")
    if not processed_clips:
        logger.warning("no clips available for merging")
        return combined_video_path

    # if there is only one clip, use it directly
    if len(processed_clips) == 1:
        logger.info("using single clip directly")
        # clip = VideoFileClip(processed_clips[0].file_path).subclipped(0, processed_clips[0].duration)
        # clip.write_videofile(
        #     filename=combined_video_path,
        #     threads=threads,
        #     logger=None,
        #     temp_audiofile_path=output_dir,
        #     audio_codec=audio_codec,
        #     fps=fps,
        # )
        # shutil.copy(processed_clips[0].file_path, combined_video_path)
        # close_clip(clip)
        input_path = processed_clips[0].file_path
        duration = processed_clips[0].duration  # in seconds
        output_path = combined_video_path

        subprocess.run([
            "ffmpeg",
            "-y",                      # overwrite output if it exists
            "-ss", "0",                # start time
            "-t", str(duration),       # duration
            "-i", input_path,          # input file
            "-r", str(fps),            # output FPS
            "-c:v", "libx264",         # video codec
            "-c:a", audio_codec,       # audio codec, e.g. "aac"
            "-threads", str(threads),  # number of threads
            "-preset", "fast",         # encoding speed (optional)
            output_path
        ])
        delete_files(processed_clips)
        logger.info("video combining completed")
        return combined_video_path

    # create initial video file as base
    base_clip_path = processed_clips[0].file_path
    temp_merged_video = f"{output_dir}/temp-merged-video.mp4"
    temp_merged_next = f"{output_dir}/temp-merged-next.mp4"

    # copy first clip as initial merged video
    shutil.copy(base_clip_path, temp_merged_video)

    # merge remaining video clips one by one
    # for i, clip in enumerate(processed_clips[1:], 1):
    #     logger.info(f"merging clip {i}/{len(processed_clips)-1}, duration: {clip.duration:.2f}s")

    #     try:
    #         # load current base video and next clip to merge
    #         base_clip = VideoFileClip(temp_merged_video)
    #         if i == len(processed_clips) - 1:
    #             next_clip = VideoFileClip(clip.file_path).subclipped(0, clip.duration)
    #         else:
    #             next_clip = VideoFileClip(clip.file_path)

    #         # merge these two clips
    #         merged_clip = concatenate_videoclips([base_clip, next_clip])

    #         # save merged result to temp file
    #         merged_clip.write_videofile(
    #             filename=temp_merged_next,
    #             threads=threads,
    #             logger=None,
    #             temp_audiofile_path=output_dir,
    #             audio_codec=audio_codec,
    #             fps=fps,
    #         )
    #         close_clip(base_clip)
    #         close_clip(next_clip)
    #         close_clip(merged_clip)

    #         # replace base file with new merged file
    #         delete_files(temp_merged_video)
    #         os.rename(temp_merged_next, temp_merged_video)

    #     except Exception as e:
    #         logger.error(f"failed to merge clip: {str(e)}")
    #         continue

    concat_list_path = os.path.join(output_dir, "concat_all.txt")

    with open(concat_list_path, "w") as f:
        for i, clip in enumerate(processed_clips):
            clip_path = clip.file_path

            # If it's the last clip and duration is set, trim it first
            if i == len(processed_clips) - 1 and clip.duration is not None:
                trimmed_path = os.path.join(output_dir, f"trimmed_last.mp4")
                subprocess.run([
                    "ffmpeg", "-y",
                    "-ss", "0",
                    "-t", str(clip.duration),
                    "-i", clip.file_path,
                    "-c:v", "libx264",
                    "-c:a", audio_codec,
                    "-threads", str(threads),
                    "-preset", "fast",
                    "-r", str(fps),
                    trimmed_path
                ])
                clip_path = trimmed_path

            logger.info(f"adding clip {i+1}/{len(processed_clips)}: {clip_path}, duration: {clip.duration:.2f}s")
            f.write(f"file '{os.path.abspath(clip_path)}'\n")

    # Run ffmpeg to concatenate all at once
    subprocess.run([
    "ffmpeg", "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", concat_list_path,
    "-c:v", "libx264",
    "-c:a", audio_codec,
    "-threads", str(threads),
    "-preset", "fast",
    "-r", str(fps),
    combined_video_path
    ])

    # Clean up
    os.remove(concat_list_path)
    if 'trimmed_path' in locals():
        os.remove(trimmed_path)


    # after merging, rename final result to target file name
    # os.rename(temp_merged_video, combined_video_path)

    # clean temp files
    clip_files = [clip.file_path for clip in processed_clips]
    delete_files(clip_files)

    logger.info("video combining completed")
    return combined_video_path


def wrap_text(text, max_width, font="Arial", fontsize=60):
    # Create ImageFont
    font = ImageFont.truetype(font, fontsize)

    def get_text_size(inner_text):
        inner_text = inner_text.strip()
        left, top, right, bottom = font.getbbox(inner_text)
        return right - left, bottom - top

    width, height = get_text_size(text)
    if width <= max_width:
        return text, height

    processed = True

    _wrapped_lines_ = []
    words = text.split(" ")
    _txt_ = ""
    for word in words:
        _before = _txt_
        _txt_ += f"{word} "
        _width, _height = get_text_size(_txt_)
        if _width <= max_width:
            continue
        else:
            if _txt_.strip() == word.strip():
                processed = False
                break
            _wrapped_lines_.append(_before)
            _txt_ = f"{word} "
    _wrapped_lines_.append(_txt_)
    if processed:
        _wrapped_lines_ = [line.strip() for line in _wrapped_lines_]
        result = "\n".join(_wrapped_lines_).strip()
        height = len(_wrapped_lines_) * height
        return result, height

    _wrapped_lines_ = []
    chars = list(text)
    _txt_ = ""
    for word in chars:
        _txt_ += word
        _width, _height = get_text_size(_txt_)
        if _width <= max_width:
            continue
        else:
            _wrapped_lines_.append(_txt_)
            _txt_ = ""
    _wrapped_lines_.append(_txt_)
    result = "\n".join(_wrapped_lines_).strip()
    height = len(_wrapped_lines_) * height
    return result, height


def generate_video(
    video_path: str,
    audio_path: str,
    subtitle_path: str,
    output_file: str,
    params: VideoParams,
):
    aspect = VideoAspect(params.video_aspect)
    video_width, video_height = aspect.to_resolution()

    logger.info(f"generating video: {video_width} x {video_height}")
    logger.info(f"  ① video: {video_path}")
    logger.info(f"  ② audio: {audio_path}")
    logger.info(f"  ③ subtitle: {subtitle_path}")
    logger.info(f"  ④ output: {output_file}")

    # https://github.com/harry0703/MoneyPrinterTurbo/issues/217
    # PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: 'final-1.mp4.tempTEMP_MPY_wvf_snd.mp3'
    # write into the same directory as the output file
    output_dir = os.path.dirname(output_file)

    font_path = ""
    if params.subtitle_enabled:
        if not params.font_name:
            params.font_name = "STHeitiMedium.ttc"
        font_path = os.path.join(utils.font_dir(), params.font_name)
        if os.name == "nt":
            font_path = font_path.replace("\\", "/")

        logger.info(f"  ⑤ font: {font_path}")

    def create_text_clip(subtitle_item):
        params.font_size = int(params.font_size)
        params.stroke_width = int(params.stroke_width)
        phrase = subtitle_item[1]
        max_width = video_width * 0.9
        wrapped_txt, txt_height = wrap_text(
            phrase, max_width=max_width, font=font_path, fontsize=params.font_size
        )
        interline = int(params.font_size * 0.25)
        size=(int(max_width), int(txt_height + params.font_size * 0.25 + (interline * (wrapped_txt.count("\n") + 1))))

        _clip = TextClip(
            text=wrapped_txt,
            font=font_path,
            font_size=params.font_size,
            color=params.text_fore_color,
            bg_color=params.text_background_color,
            stroke_color=params.stroke_color,
            stroke_width=params.stroke_width,
            # interline=interline,
            # size=size,
        )
        duration = subtitle_item[0][1] - subtitle_item[0][0]
        _clip = _clip.with_start(subtitle_item[0][0])
        _clip = _clip.with_end(subtitle_item[0][1])
        _clip = _clip.with_duration(duration)
        if params.subtitle_position == "bottom":
            _clip = _clip.with_position(("center", video_height * 0.95 - _clip.h))
        elif params.subtitle_position == "top":
            _clip = _clip.with_position(("center", video_height * 0.05))
        elif params.subtitle_position == "custom":
            # Ensure the subtitle is fully within the screen bounds
            margin = 10  # Additional margin, in pixels
            max_y = video_height - _clip.h - margin
            min_y = margin
            custom_y = (video_height - _clip.h) * (params.custom_position / 100)
            custom_y = max(
                min_y, min(custom_y, max_y)
            )  # Constrain the y value within the valid range
            _clip = _clip.with_position(("center", custom_y))
        else:  # center
            _clip = _clip.with_position(("center", "center"))
        return _clip

    video_clip = VideoFileClip(video_path).without_audio()
    audio_clip = AudioFileClip(audio_path).with_effects(
        [afx.MultiplyVolume(params.voice_volume)]
    )

    def make_textclip(text):
        return TextClip(
            text=text,
            font=font_path,
            font_size=params.font_size,
        )

    if subtitle_path and os.path.exists(subtitle_path):
        sub = SubtitlesClip(
            subtitles=subtitle_path, encoding="utf-8", make_textclip=make_textclip
        )
        text_clips = []
        for item in sub.subtitles:
            clip = create_text_clip(subtitle_item=item)
            text_clips.append(clip)
        video_clip = CompositeVideoClip([video_clip, *text_clips])

    bgm_file = get_bgm_file(bgm_type=params.bgm_type, bgm_file=params.bgm_file)
    if bgm_file:
        try:
            bgm_clip = AudioFileClip(bgm_file).with_effects(
                [
                    afx.MultiplyVolume(params.bgm_volume),
                    afx.AudioFadeOut(3),
                    afx.AudioLoop(duration=video_clip.duration),
                ]
            )
            audio_clip = CompositeAudioClip([audio_clip, bgm_clip])
        except Exception as e:
            logger.error(f"failed to add bgm: {str(e)}")

    video_clip = video_clip.with_audio(audio_clip)
    video_clip.write_videofile(
        output_file,
        audio_codec=audio_codec,
        temp_audiofile_path=output_dir,
        threads=params.n_threads or 2,
        logger=None,
        fps=fps,
    )
    video_clip.close()
    del video_clip


def preprocess_video(materials: List[MaterialInfo], clip_duration=4, image_to_video=True):
    for material in materials:
        if not material.url:
            continue

        ext = utils.parse_extension(material.url)
        try:
            clip = VideoFileClip(material.url)
        except Exception:
            clip = ImageClip(material.url)

        width = clip.size[0]
        height = clip.size[1]
        if width < 480 or height < 480:
            logger.warning(f"low resolution material: {width}x{height}, minimum 480x480 required")
            continue

        if ext in const.FILE_TYPE_IMAGES:
            logger.info(f"processing image: {material.url}")
            # Create an image clip and set its duration to 3 seconds
            clip = (
                ImageClip(material.url)
                .with_duration(clip_duration)
                .with_position("center")
            )

            if image_to_video:
                # Apply a zoom effect using the resize method.
                # A lambda function is used to make the zoom effect dynamic over time.
                # The zoom effect starts from the original size and gradually scales up to 120%.
                # t represents the current time, and clip.duration is the total duration of the clip (3 seconds).
                # Note: 1 represents 100% size, so 1.2 represents 120% size.
                zoom_clip = clip.resized(
                    lambda t: 1 + (clip_duration * 0.03) * (t / clip.duration)
                )

                # Optionally, create a composite video clip containing the zoomed clip.
                # This is useful when you want to add other elements to the video.
                final_clip = CompositeVideoClip([zoom_clip])
            else:
                # If image_to_video is False, we just use the original clip without zooming.
                final_clip = CompositeVideoClip([clip])

            # Output the video to a file.
            video_file = f"{material.url}.mp4"
            final_clip.write_videofile(video_file, fps=30, logger=None)
            close_clip(clip)
            material.url = video_file
            logger.success(f"image processed: {video_file}")
    return materials


def _is_image_file(file_path: str) -> bool:
    """
    Check if a file is an image based on its extension and optionally file content

    Args:
        file_path: Path to the file

    Returns:
        True if the file is an image, False if it's a video or other format
    """
    if not file_path:
        return False

    # Get file extension
    import os
    _, ext = os.path.splitext(file_path.lower())

    # Define image and video extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}

    # Check by extension first (most reliable)
    if ext in image_extensions:
        return True
    elif ext in video_extensions:
        return False

    # For unknown extensions or no extension, try to check if file exists and guess from content
    if os.path.exists(file_path):
        try:
            # Try to detect using file magic (if available)
            import mimetypes
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type:
                if mime_type.startswith('image/'):
                    return True
                elif mime_type.startswith('video/'):
                    return False
        except:
            pass

    # Default: assume it's an image if we can't determine (safer for backward compatibility)
    logger.warning(f"Could not determine file type for: {file_path}, defaulting to image")
    return True


def _handle_script_segments_mode(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    audio_duration: float,
    video_aspect,
    video_concat_mode,
    video_transition_mode,
    max_clip_duration: int,
    threads: int,
    params,
    nth: int
) -> str:
    """
    Handle video combination in script_segments mode with enhanced logic
    """

    script_segments = params.script_segments

    # Validate script_segments
    if nth > len(script_segments):
        raise ValueError(f"Segment index {nth} out of range. Available segments: {len(script_segments)}")

    segment = script_segments[nth - 1]

    # Strict validation of required fields
    if 'script' not in segment:
        raise ValueError(f"Segment {nth} missing required field: 'script'")
    if 'audio_duration' not in segment:
        raise ValueError(f"Segment {nth} missing required field: 'audio_duration'")
    if 'start_time' not in segment:
        raise ValueError(f"Segment {nth} missing required field: 'start_time'")

    segment_audio_duration = float(segment['audio_duration'])
    segment_start_time = float(segment['start_time'])

    if segment_audio_duration <= 0:
        raise ValueError(f"Segment {nth} audio_duration must be > 0, got: {segment_audio_duration}")
    if segment_start_time < 0:
        raise ValueError(f"Segment {nth} start_time must be >= 0, got: {segment_start_time}")

    logger.info(f"Processing segment {nth}: audio_duration={segment_audio_duration}s, start_time={segment_start_time}s")

    # Determine material type by checking actual file format
    media_path = segment.get('medias', '')
    is_image_material = _is_image_file(media_path)
    logger.info(f"Material type detection for segment {nth}: path={media_path}, is_image={is_image_material}")

    if is_image_material:
        return _handle_image_material(
            combined_video_path, video_paths, audio_file, segment_audio_duration,
            video_aspect, video_concat_mode, video_transition_mode,
            max_clip_duration, threads, segment
        )
    else:
        return _handle_video_material(
            combined_video_path, video_paths, audio_file, segment_audio_duration,
            video_aspect, video_concat_mode, video_transition_mode,
            threads, segment
        )


def _handle_image_material(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    target_duration: float,
    video_aspect,
    video_concat_mode,
    video_transition_mode,
    max_clip_duration: int,
    threads: int,
    segment: dict
) -> str:
    """
    Handle image materials with image_to_video logic
    """
    image_to_video = segment.get('image_to_video', True)

    logger.info(f"Processing image material: image_to_video={image_to_video}")

    if not image_to_video:
        # Direct display mode: create single clip with target_duration
        # Use original image path from segment instead of processed video path
        original_image_path = segment.get('medias', video_paths[0])
        return _create_single_duration_clip(
            combined_video_path, original_image_path, audio_file, target_duration,
            video_aspect, threads
        )
    else:
        # Image to video mode: use max_clip_duration and loop
        return _create_looped_clips(
            combined_video_path, video_paths, audio_file, target_duration,
            video_aspect, video_concat_mode, video_transition_mode,
            max_clip_duration, threads
        )


def _handle_video_material(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    target_duration: float,
    video_aspect,
    video_concat_mode,
    video_transition_mode,
    threads: int,
    segment: dict
) -> str:
    """
    Handle video materials with minimal loop + cutting strategy
    """
    logger.info(f"Processing video material with minimal loop + cutting strategy")

    video_path = video_paths[0]  # Assume single video file for script_segments
    segment_start_time = float(segment.get('start_time', 0))

    # Validate video file and get duration
    try:
        clip = VideoFileClip(video_path)
        video_duration = clip.duration
        close_clip(clip)
    except Exception as e:
        raise ValueError(f"Cannot read video file {video_path}: {str(e)}")

    # Validate and fix start_time
    if segment_start_time >= video_duration:
        logger.warning(f"start_time ({segment_start_time}s) >= video duration ({video_duration}s), resetting to 0")
        segment_start_time = 0

    # Calculate optimal segmentation strategy
    available_video_duration = video_duration - segment_start_time

    if target_duration <= available_video_duration:
        # Single segment: just cut from start_time to start_time + target_duration
        return _create_single_video_segment(
            combined_video_path, video_path, audio_file,
            segment_start_time, segment_start_time + target_duration,
            video_aspect, threads
        )
    else:
        # Multiple segments needed: minimal loop + cutting
        return _create_minimal_loop_segments(
            combined_video_path, video_path, audio_file, target_duration,
            segment_start_time, available_video_duration, video_duration,
            video_aspect, video_concat_mode, threads
        )


def _create_single_duration_clip(
    combined_video_path: str,
    image_path: str,
    audio_file: str,
    duration: float,
    video_aspect,
    threads: int
) -> str:
    """
    Create a single clip with specified duration (for image direct display mode)
    """

    logger.info(f"Creating single duration clip: {duration}s")

    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()

    # 预处理图片以解决imageio兼容性问题
    processed_image_path = _preprocess_image_for_imageio(image_path)

    # Create image clip with target duration
    image_clip = ImageClip(processed_image_path).with_duration(duration)

    # Resize to target resolution
    image_clip = image_clip.resized((video_width, video_height))

    # Create final composite
    final_clip = CompositeVideoClip([image_clip])

    # Add audio
    audio_clip = AudioFileClip(audio_file)
    final_clip = final_clip.with_audio(audio_clip)

    # Set fps for the final clip
    final_clip = final_clip.with_fps(24)

    # Write output
    final_clip.write_videofile(
        combined_video_path,
        threads=threads,
        logger=None,
        temp_audiofile_path=os.path.dirname(combined_video_path)
    )

    close_clip(final_clip)
    close_clip(audio_clip)

    return combined_video_path


def _create_looped_clips(
    combined_video_path: str,
    video_paths: List[str],
    audio_file: str,
    target_duration: float,
    video_aspect,
    video_concat_mode,
    video_transition_mode,
    max_clip_duration: int,
    threads: int
) -> str:
    """
    Create looped clips using the original max_clip_duration logic
    """
    logger.info(f"Using looped clips mode with max_clip_duration={max_clip_duration}s")

    # Use original logic without script_segments
    return combine_videos(
        combined_video_path, video_paths, audio_file,
        video_aspect, video_concat_mode, video_transition_mode,
        max_clip_duration, threads, None, 1
    )


def _create_single_video_segment(
    combined_video_path: str,
    video_path: str,
    audio_file: str,
    start_time: float,
    end_time: float,
    video_aspect,
    threads: int
) -> str:
    """
    Create a single video segment from start_time to end_time
    """

    duration = end_time - start_time
    logger.info(f"Creating single video segment: {start_time}s to {end_time}s (duration: {duration}s)")

    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()

    # Load and cut video
    video_clip = VideoFileClip(video_path).subclipped(start_time, end_time)

    # Resize if needed
    clip_w, clip_h = video_clip.size
    if clip_w != video_width or clip_h != video_height:
        video_clip = _resize_clip_to_aspect(video_clip, video_width, video_height)

    # Add audio
    audio_clip = AudioFileClip(audio_file)
    final_clip = video_clip.with_audio(audio_clip)

    # Write output
    final_clip.write_videofile(
        combined_video_path,
        threads=threads,
        logger=None,
        temp_audiofile_path=os.path.dirname(combined_video_path)
    )

    close_clip(final_clip)
    close_clip(audio_clip)

    return combined_video_path


def _create_minimal_loop_segments(
    combined_video_path: str,
    video_path: str,
    audio_file: str,
    target_duration: float,
    start_time: float,
    available_duration: float,
    total_video_duration: float,
    video_aspect,
    video_concat_mode,
    threads: int
) -> str:
    """
    Create minimal loop segments with 3-second minimum segment rule
    """

    MIN_SEGMENT_DURATION = 3.0

    logger.info(f"Creating minimal loop segments: target={target_duration}s, available={available_duration}s")

    # Calculate how many full cycles we need
    full_cycles = int(target_duration / available_duration)
    remaining_duration = target_duration - (full_cycles * available_duration)

    logger.info(f"Strategy: {full_cycles} full cycles + {remaining_duration:.2f}s remaining")

    # Check if remaining duration violates minimum segment rule
    if remaining_duration > 0 and remaining_duration < MIN_SEGMENT_DURATION:
        # Redistribute: create equal segments
        total_segments = full_cycles + 1
        segment_duration = target_duration / total_segments

        if segment_duration >= MIN_SEGMENT_DURATION:
            logger.info(f"Redistributing to {total_segments} equal segments of {segment_duration:.2f}s each")
            return _create_equal_segments(
                combined_video_path, video_path, audio_file, target_duration,
                segment_duration, start_time, video_aspect, threads
            )
        else:
            # Even equal segments are too short, use full cycles only
            logger.warning(f"Cannot satisfy minimum segment duration, using {full_cycles} full cycles only")
            actual_duration = full_cycles * available_duration
            return _create_full_cycles_only(
                combined_video_path, video_path, audio_file, actual_duration,
                start_time, available_duration, video_aspect, threads
            )
    else:
        # Standard approach: full cycles + remaining segment
        return _create_cycles_plus_remainder(
            combined_video_path, video_path, audio_file, target_duration,
            full_cycles, available_duration, remaining_duration,
            start_time, video_aspect, threads
        )


def _create_equal_segments(
    combined_video_path: str,
    video_path: str,
    audio_file: str,
    target_duration: float,
    segment_duration: float,
    start_time: float,
    video_aspect,
    threads: int
) -> str:
    """
    Create equal duration segments
    """

    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()

    segments_count = int(target_duration / segment_duration)
    clips = []

    for i in range(segments_count):
        segment_start = start_time
        segment_end = start_time + segment_duration

        video_clip = VideoFileClip(video_path).subclipped(segment_start, segment_end)
        video_clip = _resize_clip_to_aspect(video_clip, video_width, video_height)
        clips.append(video_clip)

    # Concatenate all clips
    final_video = concatenate_videoclips(clips)

    # Add audio
    audio_clip = AudioFileClip(audio_file)
    final_clip = final_video.with_audio(audio_clip)

    # Write output
    final_clip.write_videofile(
        combined_video_path,
        threads=threads,
        logger=None,
        temp_audiofile_path=os.path.dirname(combined_video_path)
    )

    # Cleanup
    for clip in clips:
        close_clip(clip)
    close_clip(final_clip)
    close_clip(audio_clip)

    return combined_video_path


def _create_full_cycles_only(
    combined_video_path: str,
    video_path: str,
    audio_file: str,
    actual_duration: float,
    start_time: float,
    cycle_duration: float,
    video_aspect,
    threads: int
) -> str:
    """
    Create video with full cycles only (no remainder)
    """
    cycles_count = int(actual_duration / cycle_duration)
    return _create_cycles_plus_remainder(
        combined_video_path, video_path, audio_file, actual_duration,
        cycles_count, cycle_duration, 0,
        start_time, video_aspect, threads
    )


def _create_cycles_plus_remainder(
    combined_video_path: str,
    video_path: str,
    audio_file: str,
    target_duration: float,
    full_cycles: int,
    cycle_duration: float,
    remaining_duration: float,
    start_time: float,
    video_aspect,
    threads: int
) -> str:
    """
    Create video with full cycles plus remainder segment
    """

    aspect = VideoAspect(video_aspect)
    video_width, video_height = aspect.to_resolution()

    clips = []

    # Add full cycles
    for i in range(full_cycles):
        cycle_start = start_time
        cycle_end = start_time + cycle_duration

        video_clip = VideoFileClip(video_path).subclipped(cycle_start, cycle_end)
        video_clip = _resize_clip_to_aspect(video_clip, video_width, video_height)
        clips.append(video_clip)

    # Add remainder segment if needed
    if remaining_duration > 0:
        remainder_start = start_time
        remainder_end = start_time + remaining_duration

        video_clip = VideoFileClip(video_path).subclipped(remainder_start, remainder_end)
        video_clip = _resize_clip_to_aspect(video_clip, video_width, video_height)
        clips.append(video_clip)

    # Concatenate all clips
    final_video = concatenate_videoclips(clips)

    # Add audio
    audio_clip = AudioFileClip(audio_file)
    final_clip = final_video.with_audio(audio_clip)

    # Write output
    final_clip.write_videofile(
        combined_video_path,
        threads=threads,
        logger=None,
        temp_audiofile_path=os.path.dirname(combined_video_path)
    )

    # Cleanup
    for clip in clips:
        close_clip(clip)
    close_clip(final_clip)
    close_clip(audio_clip)

    return combined_video_path


def _resize_clip_to_aspect(clip, target_width: int, target_height: int):
    """
    Resize clip to target aspect ratio with proper scaling
    """

    clip_w, clip_h = clip.size

    if clip_w == target_width and clip_h == target_height:
        return clip

    clip_ratio = clip_w / clip_h
    target_ratio = target_width / target_height

    if clip_ratio == target_ratio:
        return clip.resized((target_width, target_height))
    else:
        # Different aspect ratios, use letterboxing
        if clip_ratio > target_ratio:
            scale_factor = target_width / clip_w
        else:
            scale_factor = target_height / clip_h

        new_width = int(clip_w * scale_factor)
        new_height = int(clip_h * scale_factor)

        background = ColorClip(size=(target_width, target_height), color=(0, 0, 0)).with_duration(clip.duration)
        clip_resized = clip.resized((new_width, new_height)).with_position("center")
        return CompositeVideoClip([background, clip_resized])
