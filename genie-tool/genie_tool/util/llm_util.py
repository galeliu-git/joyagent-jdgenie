'''
Author: galeliu
Date: 2025-10-09 17:12:10
LastEditTime: 2025-10-10 16:27:06
LastEditors: galeliu
Description: .
'''
# -*- coding: utf-8 -*-
# =====================
# 
# 
# Author: liumin.423
# Date:   2025/7/8
# =====================
import json
import os
from typing import List, Any, Optional

from litellm import acompletion

from genie_tool.util.log_util import timer, AsyncTimer
from genie_tool.util.sensitive_detection import SensitiveWordsReplace


@timer(key="enter")
async def ask_llm(
        messages: str | List[Any],
        model: str,
        temperature: float = None,
        top_p: float = None,
        stream: bool = False,

        # 自定义字段
        only_content: bool = False,     # 只返回内容

        extra_headers: Optional[dict] = None,
        **kwargs,
):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if os.getenv("SENSITIVE_WORD_REPLACE", "false") == "true":
        for message in messages:
            if isinstance(message.get("content"), str):
                message["content"] = SensitiveWordsReplace.replace(message["content"])
            else:
                message["content"] = json.loads(
                    SensitiveWordsReplace.replace(json.dumps(message["content"], ensure_ascii=False)))
    # ✅ 参数修正，兼容 DeepSeek / OpenAI
    if 'deepseek' in model:
        if top_p is None or top_p <= 0 or top_p > 1:
            top_p = 0.9 # DeepSeek 要求 (0, 1.0]
        if temperature is None:
            temperature = 0.7  # 保底温度
    print(f'top_p:{top_p},temperature:{temperature}')
    response = await acompletion(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
        extra_headers=extra_headers,
        **kwargs
    )
    async with AsyncTimer(key=f"exec ask_llm"):
        if stream:
            async for chunk in response:
                if only_content:
                    if chunk.choices and chunk.choices[0] and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                else:
                    yield chunk
        else:
            yield response.choices[0].message.content if only_content else response


if __name__ == "__main__":
    pass
