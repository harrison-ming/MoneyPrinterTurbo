import json
import logging
import re
from typing import List

import g4f
from loguru import logger
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.exceptions import LLMResponseValidationError
from typing import List, Dict, Any

from app.config import config

_max_retries = 5


def _generate_response(prompt: str) -> str:
    try:
        content = ""
        llm_provider = config.app.get("llm_provider", "openai")
        logger.info(f"llm provider: {llm_provider}")
        if llm_provider == "g4f":
            model_name = config.app.get("g4f_model_name", "")
            if not model_name:
                model_name = "gpt-3.5-turbo-16k-0613"
            content = g4f.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
        else:
            api_version = ""  # for azure
            if llm_provider == "moonshot":
                api_key = config.app.get("moonshot_api_key")
                model_name = config.app.get("moonshot_model_name")
                base_url = "https://api.moonshot.cn/v1"
            elif llm_provider == "ollama":
                # api_key = config.app.get("openai_api_key")
                api_key = "ollama"  # any string works but you are required to have one
                model_name = config.app.get("ollama_model_name")
                base_url = config.app.get("ollama_base_url", "")
                if not base_url:
                    base_url = "http://localhost:11434/v1"
            elif llm_provider == "openai":
                api_key = config.app.get("openai_api_key")
                model_name = config.app.get("openai_model_name")
                base_url = config.app.get("openai_base_url", "")
                if not base_url:
                    base_url = "https://api.openai.com/v1"
            elif llm_provider == "oneapi":
                api_key = config.app.get("oneapi_api_key")
                model_name = config.app.get("oneapi_model_name")
                base_url = config.app.get("oneapi_base_url", "")
            elif llm_provider == "azure":
                api_key = config.app.get("azure_api_key")
                model_name = config.app.get("azure_model_name")
                base_url = config.app.get("azure_base_url", "")
                api_version = config.app.get("azure_api_version", "2024-02-15-preview")
            elif llm_provider == "gemini":
                api_key = config.app.get("gemini_api_key")
                model_name = config.app.get("gemini_model_name")
                base_url = "***"
            elif llm_provider == "qwen":
                api_key = config.app.get("qwen_api_key")
                model_name = config.app.get("qwen_model_name")
                base_url = "***"
            elif llm_provider == "cloudflare":
                api_key = config.app.get("cloudflare_api_key")
                model_name = config.app.get("cloudflare_model_name")
                account_id = config.app.get("cloudflare_account_id")
                base_url = "***"
            elif llm_provider == "deepseek":
                api_key = config.app.get("deepseek_api_key")
                model_name = config.app.get("deepseek_model_name")
                base_url = config.app.get("deepseek_base_url")
                if not base_url:
                    base_url = "https://api.deepseek.com"
            elif llm_provider == "ernie":
                api_key = config.app.get("ernie_api_key")
                secret_key = config.app.get("ernie_secret_key")
                base_url = config.app.get("ernie_base_url")
                model_name = "***"
                if not secret_key:
                    raise ValueError(
                        f"{llm_provider}: secret_key is not set, please set it in the config.toml file."
                    )
            else:
                raise ValueError(
                    "llm_provider is not set, please set it in the config.toml file."
                )

            if not api_key:
                raise ValueError(
                    f"{llm_provider}: api_key is not set, please set it in the config.toml file."
                )
            if not model_name:
                raise ValueError(
                    f"{llm_provider}: model_name is not set, please set it in the config.toml file."
                )
            if not base_url:
                raise ValueError(
                    f"{llm_provider}: base_url is not set, please set it in the config.toml file."
                )

            if llm_provider == "qwen":
                import dashscope
                from dashscope.api_entities.dashscope_response import GenerationResponse

                dashscope.api_key = api_key
                response = dashscope.Generation.call(
                    model=model_name, messages=[{"role": "user", "content": prompt}]
                )
                if response:
                    if isinstance(response, GenerationResponse):
                        status_code = response.status_code
                        if status_code != 200:
                            raise Exception(
                                f'[{llm_provider}] returned an error response: "{response}"'
                            )

                        content = response["output"]["text"]
                        return content.replace("\n", "")
                    else:
                        raise Exception(
                            f'[{llm_provider}] returned an invalid response: "{response}"'
                        )
                else:
                    raise Exception(f"[{llm_provider}] returned an empty response")

            if llm_provider == "gemini":
                import google.generativeai as genai

                genai.configure(api_key=api_key, transport="rest")

                generation_config = {
                    "temperature": 0.5,
                    "top_p": 1,
                    "top_k": 1,
                    "max_output_tokens": 2048,
                }

                safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_ONLY_HIGH",
                    },
                ]

                model = genai.GenerativeModel(
                    model_name=model_name,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )

                try:
                    response = model.generate_content(prompt)
                    candidates = response.candidates
                    generated_text = candidates[0].content.parts[0].text
                except (AttributeError, IndexError) as e:
                    print("Gemini Error:", e)

                return generated_text

            if llm_provider == "cloudflare":
                import requests

                response = requests.post(
                    f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_name}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a friendly assistant",
                            },
                            {"role": "user", "content": prompt},
                        ]
                    },
                )
                result = response.json()
                logger.info(result)
                return result["result"]["response"]

            if llm_provider == "ernie":
                import requests

                params = {
                    "grant_type": "client_credentials",
                    "client_id": api_key,
                    "client_secret": secret_key,
                }
                access_token = (
                    requests.post(
                        "https://aip.baidubce.com/oauth/2.0/token", params=params
                    )
                    .json()
                    .get("access_token")
                )
                url = f"{base_url}?access_token={access_token}"

                payload = json.dumps(
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.5,
                        "top_p": 0.8,
                        "penalty_score": 1,
                        "disable_search": False,
                        "enable_citation": False,
                        "response_format": "text",
                    }
                )
                headers = {"Content-Type": "application/json"}

                response = requests.request(
                    "POST", url, headers=headers, data=payload
                ).json()
                return response.get("result")

            if llm_provider == "azure":
                client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=base_url,
                )
            else:
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                )

            response = client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": prompt}]
            )
            if response:
                if isinstance(response, ChatCompletion):
                    content = response.choices[0].message.content
                else:
                    raise Exception(
                        f'[{llm_provider}] returned an invalid response: "{response}", please check your network '
                        f"connection and try again."
                    )
            else:
                raise Exception(
                    f"[{llm_provider}] returned an empty response, please check your network connection and try again."
                )

        return content.replace("\n", "")
    except Exception as e:
        return f"Error: {str(e)}"


def _generate_story_response(messages: List[Dict[str, str]], response_format: str = "json_object") -> any:
        """生成 LLM 响应

        Args:
            messages: 消息列表
            response_format: 响应格式，默认为 json_object

        Returns:
            Dict[str, Any]: 解析后的响应

        Raises:
            Exception: 请求失败或解析失败时抛出异常
        """
        llm_provider = config.app.get("llm_provider", "openai")
        logger.info(f"llm provider: {llm_provider}")
        api_key = config.app.get("openai_api_key")
        model_name = config.app.get("openai_model_name")
        base_url = config.app.get("openai_base_url", "")
        if not base_url:
            base_url = "https://api.openai.com/v1"

        if not api_key:
            raise ValueError(
                f"{llm_provider}: api_key is not set, please set it in the config.toml file."
            )
        if not model_name:
            raise ValueError(
                f"{llm_provider}: model_name is not set, please set it in the config.toml file."
            )
        if not base_url:
            raise ValueError(
                f"{llm_provider}: base_url is not set, please set it in the config.toml file."
            )
            
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        response = client.chat.completions.create(
            model= model_name,
            response_format={"type": response_format},
            messages=messages,
        )
        try:
            content = response.choices[0].message.content
            result = json.loads(content)
            return result
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            raise e
        

def generate_script(
    video_subject: str, language: str = "", paragraph_number: int = 1
) -> str:
    prompt = f"""
# Role: Video Script Generator

## Goals:
Generate a script for a video, depending on the subject of the video.

## Constrains:
1. the script is to be returned as a string with the specified number of paragraphs.
2. do not under any circumstance reference this prompt in your response.
3. get straight to the point, don't start with unnecessary things like, "welcome to this video".
4. you must not include any type of markdown or formatting in the script, never use a title.
5. only return the raw content of the script.
6. do not include "voiceover", "narrator" or similar indicators of what should be spoken at the beginning of each paragraph or line.
7. you must not mention the prompt, or anything about the script itself. also, never talk about the amount of paragraphs or lines. just write the script.
8. respond in the same language as the video subject.

# Initialization:
- video subject: {video_subject}
- number of paragraphs: {paragraph_number}
""".strip()
    if language:
        prompt += f"\n- language: {language}"

    final_script = ""
    logger.info(f"subject: {video_subject}")

    def format_response(response):
        # Clean the script
        # Remove asterisks, hashes
        response = response.replace("*", "")
        response = response.replace("#", "")

        # Remove markdown syntax
        response = re.sub(r"\[.*\]", "", response)
        response = re.sub(r"\(.*\)", "", response)

        # Split the script into paragraphs
        paragraphs = response.split("\n\n")

        # Select the specified number of paragraphs
        # selected_paragraphs = paragraphs[:paragraph_number]

        # Join the selected paragraphs into a single string
        return "\n\n".join(paragraphs)

    for i in range(_max_retries):
        try:
            response = _generate_response(prompt=prompt)
            if response:
                final_script = format_response(response)
            else:
                logging.error("gpt returned an empty response")

            # g4f may return an error message
            if final_script and "当日额度已消耗完" in final_script:
                raise ValueError(final_script)

            if final_script:
                break
        except Exception as e:
            logger.error(f"failed to generate script: {e}")

        if i < _max_retries:
            logger.warning(f"failed to generate video script, trying again... {i + 1}")
    if "Error: " in final_script:
        logger.error(f"failed to generate video script: {final_script}")
    else:
        logger.success(f"completed: \n{final_script}")
    return final_script.strip()


def generate_terms(video_subject: str, video_script: str, amount: int = 5) -> List[str]:
    prompt = f"""
# Role: Video Search Terms Generator

## Goals:
Generate {amount} search terms for stock videos, depending on the subject of a video.

## Constrains:
1. the search terms are to be returned as a json-array of strings.
2. each search term should consist of 1-3 words, always add the main subject of the video.
3. you must only return the json-array of strings. you must not return anything else. you must not return the script.
4. the search terms must be related to the subject of the video.
5. reply with english search terms only.

## Output Example:
["search term 1", "search term 2", "search term 3","search term 4","search term 5"]

## Context:
### Video Subject
{video_subject}

### Video Script
{video_script}

Please note that you must use English for generating video search terms; Chinese is not accepted.
""".strip()

    logger.info(f"subject: {video_subject}")

    search_terms = []
    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if "Error: " in response:
                logger.error(f"failed to generate video script: {response}")
                return response
            search_terms = json.loads(response)
            if not isinstance(search_terms, list) or not all(
                isinstance(term, str) for term in search_terms
            ):
                logger.error("response is not a list of strings.")
                continue

        except Exception as e:
            logger.warning(f"failed to generate video terms: {str(e)}")
            if response:
                match = re.search(r"\[.*]", response)
                if match:
                    try:
                        search_terms = json.loads(match.group())
                    except Exception as e:
                        logger.warning(f"failed to generate video terms: {str(e)}")
                        pass

        if search_terms and len(search_terms) > 0:
            break
        if i < _max_retries:
            logger.warning(f"failed to generate video terms, trying again... {i + 1}")

    logger.success(f"completed: \n{search_terms}")
    return search_terms


def generate_story(story: str, language: str, segments: int) -> List[Dict[str, Any]]:
        """生成故事场景
        Args:
            story_prompt (str, optional): 故事提示. Defaults to None.
            segments (int, optional): 故事分段数. Defaults to 3.

        Returns:
            List[Dict[str, Any]]: 故事场景列表
        """
        
        messages = [
            {"role": "system", "content": "你是一个专业的故事创作者，善于创作引人入胜的故事。请只返回JSON格式的内容。"},
            {"role": "user", "content": _get_story_prompt(story, language, segments)}
        ]
        logger.info(f"prompt messages: {json.dumps(messages, indent=4, ensure_ascii=False)}")
        response = _generate_story_response(messages=messages, response_format="json_object")
        response = response["list"]
        response = normalize_keys(response)

        logger.info(f"Generated story: {json.dumps(response, indent=4, ensure_ascii=False)}")
        # 验证响应格式
        _validate_story_response(response)
        
        return response


def normalize_keys(data):
        """
        阿里云和 openai 的模型返回结果不一致，处理一下
        修改对象中非 `text` 的键为 `image_prompt`
        - 如果是字典，替换 `text` 以外的单个键为 `image_prompt`
        - 如果是列表，对列表中的每个对象递归处理
        """
        if isinstance(data, dict):
            # 如果是字典，处理键值
            if "text" in data:
                # 找到非 `text` 的键
                other_keys = [key for key in data.keys() if key != "text"]
                # 确保只处理一个非 `text` 键的情况
                if len(other_keys) == 1:
                    data["image_prompt"] = data.pop(other_keys[0])
                elif len(other_keys) > 1:
                    raise ValueError(f"Unexpected extra keys: {other_keys}. Only one non-'text' key is allowed.")
            return data
        elif isinstance(data, list):
            # 如果是列表，递归处理每个对象
            return [normalize_keys(item) for item in data]
        else:
            raise TypeError("Input must be a dict or list of dicts")


def generate_image(prompt: str, resolution: str = "1024x1024") -> str:
        # return "https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/1d/56/20250118/3c4cc727/4fc622b5-54a6-484c-bf1f-f1cfb66ace2d-1.png?Expires=1737290655&OSSAccessKeyId=LTAI5tQZd8AEcZX6KZV4G8qL&Signature=W8D4CN3uonQ2pL1e9xGMWufz33E%3D"
        """生成图片

        Args:
            prompt (str): 图片描述
            resolution (str): 图片分辨率，默认为 1024x1024

        Returns:
            str: 图片URL
        """


        try:
            # 添加安全提示词
            safe_prompt = f"Create a safe, family-friendly illustration. {prompt} The image should be appropriate for all ages, non-violent, and non-controversial."
            if (resolution != None):
                resolution = resolution.replace("*", "x")

            llm_provider = config.app.get("llm_provider", "openai")
            image_llm_model = config.app.get("image_llm_model", "dall-e-3")
            logger.info(f"llm provider: {llm_provider}")
            api_key = config.app.get("openai_api_key")
            model_name = config.app.get("openai_model_name")
            base_url = config.app.get("openai_base_url", "")
            if not base_url:
                base_url = "https://api.openai.com/v1"

            if not api_key:
                raise ValueError(
                    f"{llm_provider}: api_key is not set, please set it in the config.toml file."
                )
            if not model_name:
                raise ValueError(
                    f"{llm_provider}: model_name is not set, please set it in the config.toml file."
                )
            if not base_url:
                raise ValueError(
                    f"{llm_provider}: base_url is not set, please set it in the config.toml file."
                )
                
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            response = client.images.generate(
                model=image_llm_model,
                prompt=safe_prompt,
                size=resolution,
                quality="standard",
                n=1
            )
            logger.info("image generate res", response.data[0].url)
            return response.data[0].url
        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            return ""


def generate_story_with_images(story: str, language: str, segments: int, resolution: str) -> List[Dict[str, Any]]:
        """生成故事和配图
        Args:
            story (str, optional): 故事. Defaults to None.
            language (Language, optional): 语言. Defaults to Language.CHINESE.
            segments (int, optional): 故事分段数. Defaults to 3.
            resolution (str, optional): 图片分辨率. Defaults to "1024x1024".

        Returns:
            List[Dict[str, Any]]: 故事场景列表，每个场景包含文本、图片提示词和图片URL
        """
        # 先生成故事
        story_segments = generate_story(
            story=story, language=language, segments=segments
        )

        # 为每个场景生成图片
        for segment in story_segments:
            try:
                image_url = generate_image(prompt=segment["image_prompt"], resolution=resolution)
                segment["url"] = image_url
            except Exception as e:
                logger.error(f"Failed to generate image for segment: {e}")
                segment["url"] = None

        return story_segments


def _validate_story_response(response: any) -> None:
        """验证故事生成响应

        Args:
            response: LLM 响应

        Raises:
            LLMResponseValidationError: 响应格式错误
        """
        if not isinstance(response, list):
            raise LLMResponseValidationError("Response must be an array")

        for i, scene in enumerate(response):
            if not isinstance(scene, dict):
                raise LLMResponseValidationError(f"story item {i} must be an object")
            
            if "text" not in scene:
                raise LLMResponseValidationError(f"Scene {i} missing 'text' field")
            
            if "image_prompt" not in scene:
                raise LLMResponseValidationError(f"Scene {i} missing 'image_prompt' field")
            
            if not isinstance(scene["text"], str):
                raise LLMResponseValidationError(f"Scene {i} 'text' must be a string")
            
            if not isinstance(scene["image_prompt"], str):
                raise LLMResponseValidationError(f"Scene {i} 'image_prompt' must be a string")


def _get_story_prompt(story: str = None, language: str = "中文（简体）", segments: int = 3) -> str:
        """生成故事提示词

        Args:
            story (str, optional): 故事. Defaults to None.
            segments (int, optional): 故事分段数. Defaults to 3.

        Returns:
            str: 完整的提示词
        """

        # if story_prompt:
        base_prompt = f"有以下故事：{story}"
        
        return f"""
        {base_prompt}. The story needs to be divided into {segments} scenes, and each scene must include descriptive text and an image prompt.

        Please return the result in the following JSON format, where the key `list` contains an array of objects:

        **Expected JSON format**:
        {{
            "list": [
                {{
                    "text": "Descriptive text for the scene",
                    "image_prompt": "Detailed image generation prompt, described in English"
                }},
                {{
                    "text": "Another scene description text",
                    "image_prompt": "Another detailed image generation prompt in English"
                }}
            ]
        }}

        **Requirements**:
        1. The root object must contain a key named `list`, and its value must be an array of scene objects.
        2. Each object in the `list` array must include:
            - `text`: A descriptive text for the scene, written in {language}.
            - `image_prompt`: A detailed prompt for generating an image, written in English.
        3. Ensure the JSON format matches the above example exactly. Avoid extra fields or incorrect key names like `cimage_prompt` or `inage_prompt`.

        **Important**:
        - If there is only one scene, the array under `list` should contain a single object.
        - The output must be a valid JSON object. Do not include explanations, comments, or additional content outside the JSON.

        Example output:
        {{
            "list": [
                {{
                    "text": "Scene description text",
                    "image_prompt": "Detailed image generation prompt in English"
                }}
            ]
        }}
        """


if __name__ == "__main__":
    video_subject = "生命的意义是什么"
    script = generate_script(
        video_subject=video_subject, language="zh-CN", paragraph_number=1
    )
    print("######################")
    print(script)
    search_terms = generate_terms(
        video_subject=video_subject, video_script=script, amount=5
    )
    print("######################")
    print(search_terms)
