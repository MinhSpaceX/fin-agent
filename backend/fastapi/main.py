import os
import uuid
import time
import json
from typing import List, Dict, Any, Optional, Union, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Configuration ---
LANGGRAPH_API_BASE_URL = os.getenv("LANGGRAPH_API_BASE_URL", "http://0.0.0.0:2024")
PROXY_API_KEY = os.getenv("PROXY_API_KEY", None) 
LANGGRAPH_ASSISTANT_SEARCH_GRAPH_ID = os.getenv("LANGGRAPH_ASSISTANT_SEARCH_GRAPH_ID")


# --- Pydantic Models for OpenAI Compatibility ---

class ModelCard(BaseModel):
    id: str 
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "langgraph_proxy"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str 
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None 
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]

class ChatCompletionStreamChoiceDelta(BaseModel):
    role: Optional[Literal["system", "user", "assistant", "tool"]] = None
    content: Optional[str] = None

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: ChatCompletionStreamChoiceDelta 
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str 
    object: str = "chat.completion.chunk"
    created: int 
    model: str
    choices: List[ChatCompletionStreamChoice]

# --- FastAPI Application ---
app = FastAPI(
    title="OpenAI to LangGraph Proxy (Stateful)",
    version="0.2.7", # Incremented version
    description="A proxy to make LangGraph Platform API compatible with OpenAI API style, with refined stream handling to prevent duplicates.",
)

origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, 
    allow_methods=["*"],    
    allow_headers=["*"],    
)

async_client = httpx.AsyncClient(timeout=180.0) 

MODEL_NAME_TO_ASSISTANT_ID_MAPPING: Dict[str, str] = {}
ACTIVE_THREADS_PER_MODEL: Dict[str, str] = {} # Key: assistant_id_uuid (UUID), Value: thread_id


@app.on_event("startup")
async def startup_event():
    await _update_model_mapping()


async def _update_model_mapping():
    global MODEL_NAME_TO_ASSISTANT_ID_MAPPING
    langgraph_assistants_url = f"{LANGGRAPH_API_BASE_URL}/assistants/search"
    payload: Dict[str, Any] = {"metadata": {}, "limit": 1000, "offset": 0}
    raw_search_graph_id = os.getenv("LANGGRAPH_ASSISTANT_SEARCH_GRAPH_ID")
    if raw_search_graph_id is not None:
        if raw_search_graph_id != "": payload["graph_id"] = raw_search_graph_id
    else: payload["graph_id"] = "agent"
    
    print(f"DEBUG: _update_model_mapping - Using payload for /assistants/search: {json.dumps(payload)}")
    temp_mapping = {}
    try:
        response = await async_client.post(langgraph_assistants_url, json=payload)
        response.raise_for_status()
        assistants_data = response.json()
        if isinstance(assistants_data, list):
            for assistant in assistants_data:
                assistant_id_uuid = assistant.get("assistant_id")
                name = assistant.get("name")
                display_id = name if name and name.strip() else assistant_id_uuid
                if assistant_id_uuid and display_id: 
                    temp_mapping[display_id] = assistant_id_uuid 
            MODEL_NAME_TO_ASSISTANT_ID_MAPPING = temp_mapping 
            print(f"DEBUG: Updated model name to assistant_id mapping: {MODEL_NAME_TO_ASSISTANT_ID_MAPPING}")
    except Exception as e:
        print(f"ERROR: Failed to update model name to assistant_id mapping: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    await async_client.aclose()

async def check_proxy_api_key(authorization: Optional[str] = Header(None)):
    if PROXY_API_KEY:
        if authorization is None: raise HTTPException(status_code=401, detail="Not authenticated (Missing Authorization header)")
        scheme, _, credentials = authorization.partition(" ")
        if scheme.lower() != "bearer" or credentials != PROXY_API_KEY: raise HTTPException(status_code=403, detail="Forbidden (Invalid API Key)")

@app.get("/v1/models", response_model=ModelList)
async def list_models(authorization: Optional[str] = Header(None)):
    await check_proxy_api_key(authorization)
    await _update_model_mapping() 
    models = [ModelCard(id=display_id) for display_id in MODEL_NAME_TO_ASSISTANT_ID_MAPPING.keys()]
    if not models: print("WARNING: /v1/models - No models found.")
    return ModelList(data=models)
        

async def get_or_create_thread(assistant_id_uuid: str, force_new: bool = False) -> str:
    global ACTIVE_THREADS_PER_MODEL
    if not force_new:
        thread_id = ACTIVE_THREADS_PER_MODEL.get(assistant_id_uuid)
        if thread_id:
            print(f"DEBUG: Using existing thread_id '{thread_id}' for assistant_id_uuid '{assistant_id_uuid}'")
            return thread_id
    action = "Creating new (forced by logic or retry)" if force_new else "No active thread found. Creating new"
    print(f"DEBUG: {action} thread for assistant_id_uuid '{assistant_id_uuid}'.")
    create_thread_url = f"{LANGGRAPH_API_BASE_URL}/threads"
    try:
        response = await async_client.post(create_thread_url, json={}) 
        response.raise_for_status()
        thread_data = response.json()
        new_thread_id = thread_data.get("thread_id")
        if not new_thread_id:
            print(f"ERROR: LangGraph /threads did not return a thread_id. Response: {thread_data}")
            raise HTTPException(status_code=500, detail="Failed to create or retrieve thread ID from LangGraph.")
        ACTIVE_THREADS_PER_MODEL[assistant_id_uuid] = new_thread_id 
        print(f"DEBUG: Created new thread_id '{new_thread_id}' for assistant_id_uuid '{assistant_id_uuid}' and stored it.")
        return new_thread_id
    except httpx.HTTPStatusError as e:
        print(f"ERROR: Failed to create LangGraph thread for assistant_id_uuid '{assistant_id_uuid}'. Status: {e.response.status_code}, Detail: {e.response.text}")
        raise HTTPException(status_code=500, detail=f"Failed to create LangGraph thread: {e.response.text}")
    except Exception as e:
        print(f"ERROR: Exception while creating LangGraph thread for assistant_id_uuid '{assistant_id_uuid}': {e}")
        raise HTTPException(status_code=500, detail=f"Exception creating LangGraph thread: {str(e)}")


@app.post("/v1/chat/completions")
async def chat_completions(
    request_data: ChatCompletionRequest, 
    request: Request, 
    authorization: Optional[str] = Header(None)
):
    await check_proxy_api_key(authorization)

    model_display_name = request_data.model 
    assistant_id_uuid = MODEL_NAME_TO_ASSISTANT_ID_MAPPING.get(model_display_name)

    if not assistant_id_uuid:
        await _update_model_mapping() 
        assistant_id_uuid = MODEL_NAME_TO_ASSISTANT_ID_MAPPING.get(model_display_name)
        if not assistant_id_uuid:
            raise HTTPException(status_code=404, detail=f"Model '{model_display_name}' not found in proxy mapping.")
    
    run_identifier = model_display_name 
    completion_id = f"chatcmpl-{uuid.uuid4().hex}" 
    request_created_time = int(time.time())
    langgraph_input = {"messages": [msg.model_dump(exclude_none=True) for msg in request_data.messages]}
    langgraph_config_root: Dict[str, Any] = {} 
    configurable_params: Dict[str, Any] = {}

    if request_data.temperature is not None: configurable_params["temperature"] = request_data.temperature
    if request_data.max_tokens is not None: configurable_params["max_tokens"] = request_data.max_tokens 
    if request_data.top_p is not None: configurable_params["top_p"] = request_data.top_p
    if request_data.stop is not None: configurable_params["stop"] = request_data.stop
    if configurable_params: langgraph_config_root["configurable"] = configurable_params

    langgraph_stream_modes = ["messages-tuple", "values"] 
    langgraph_payload = {
        "assistant_id": run_identifier, 
        "input": langgraph_input,
        "stream_mode": langgraph_stream_modes 
    }
    if langgraph_config_root: 
        langgraph_payload["config"] = langgraph_config_root

    is_new_chat_heuristic = len(request_data.messages) == 1

    for attempt in range(2): 
        force_new_thread_for_this_attempt = is_new_chat_heuristic or (attempt > 0)
        if is_new_chat_heuristic and attempt == 0:
             print(f"INFO: Detected potential new chat for model '{model_display_name}'. Forcing new thread.")
        
        thread_id = await get_or_create_thread(assistant_id_uuid, force_new=force_new_thread_for_this_attempt)
        print(f"DEBUG: Chat completion attempt {attempt + 1} for model '{model_display_name}' (assistant_id_uuid: '{assistant_id_uuid}') using thread_id: '{thread_id}'. Using '{run_identifier}' as assistant_id in run payload.")

        if request_data.stream:
            langgraph_run_url = f"{LANGGRAPH_API_BASE_URL}/threads/{thread_id}/runs/stream"
            async def stream_generator():
                has_sent_content_for_turn = False # Renamed for clarity
                current_turn_finish_reason: Optional[str] = None
                non_yielding_error = None
                try:
                    print(f"DEBUG: Attempting to stream from LangGraph (Threaded): {langgraph_run_url} with payload: {json.dumps(langgraph_payload)}")
                    async with async_client.stream("POST", langgraph_run_url, json=langgraph_payload) as response:
                        if response.status_code != 200:
                            error_content = await response.aread()
                            error_detail_str = f"LangGraph API error ({response.status_code}) for run on thread '{thread_id}' with assistant '{run_identifier}'"
                            try: error_json = json.loads(error_content.decode()); error_detail_str += f": {error_json.get('detail', str(error_json))}"
                            except: error_detail_str += f" - {error_content.decode()}"
                            print(f"ERROR: {error_detail_str}") 
                            if response.status_code == 404: non_yielding_error = httpx.HTTPStatusError(error_detail_str, request=response.request, response=response); return
                            final_error_delta = ChatCompletionStreamChoiceDelta(role="assistant", content=f"Error: {error_detail_str[:1000]}")
                            error_choice = ChatCompletionStreamChoice(index=0, delta=final_error_delta, finish_reason="error")
                            yield f"data: {ChatCompletionStreamResponse(id=completion_id, created=request_created_time, model=model_display_name, choices=[error_choice]).model_dump_json(exclude_none=True)}\n\n"
                            yield "data: [DONE]\n\n"; return
                        
                        async for line in response.aiter_lines():
                            if line.startswith("data:"):
                                try:
                                    data_json_str = line[len("data:"):].strip();
                                    if not data_json_str or data_json_str == "[DONE]": continue 
                                    parsed_event_data = json.loads(data_json_str)
                                    # print(f"DEBUG: Raw LangGraph event data: {json.dumps(parsed_event_data)}") # Can be very verbose

                                    events_to_process = parsed_event_data if isinstance(parsed_event_data, list) else [parsed_event_data]

                                    for langgraph_event_item in events_to_process:
                                        if not isinstance(langgraph_event_item, dict): 
                                            print(f"WARNING: Skipping non-dictionary item in LangGraph event list: {langgraph_event_item}")
                                            continue

                                        print(f"DEBUG: Processing LangGraph event item (Threaded, mode: {langgraph_stream_modes}): {json.dumps(langgraph_event_item)}") 
                                        
                                        # Default for this item
                                        item_content_chunk: Optional[str] = None
                                        item_finish_reason: Optional[str] = None
                                        is_ai_message_chunk = False
                                        is_final_messages_event = False

                                        if langgraph_event_item.get("type") == "AIMessageChunk":
                                            is_ai_message_chunk = True
                                            item_content_chunk = langgraph_event_item.get("content")
                                            response_metadata = langgraph_event_item.get("response_metadata")
                                            if isinstance(response_metadata, dict):
                                                fr_meta = response_metadata.get("finish_reason")
                                                if fr_meta: item_finish_reason = fr_meta.lower() if fr_meta.upper() not in ["STOP", "LENGTH"] else fr_meta.upper()
                                        
                                        elif "messages" in langgraph_event_item and isinstance(langgraph_event_item["messages"], list):
                                            is_final_messages_event = True # Assume this structure contains the final state for the turn
                                            if langgraph_event_item["messages"]:
                                                last_message = langgraph_event_item["messages"][-1]
                                                if isinstance(last_message, dict) and (last_message.get("type") == "ai" or last_message.get("role") == "assistant"): 
                                                    # Only use content from "messages" if no AIMessageChunks were processed for this turn
                                                    if not has_sent_content_for_turn:
                                                        item_content_chunk = last_message.get("content")
                                                    
                                                    response_metadata = last_message.get("response_metadata")
                                                    if isinstance(response_metadata, dict):
                                                        fr_meta = response_metadata.get("finish_reason")
                                                        if fr_meta: item_finish_reason = fr_meta.lower() if fr_meta.upper() not in ["STOP", "LENGTH"] else fr_meta.upper()
                                        
                                        elif isinstance(langgraph_event_item, dict) and "values" in langgraph_event_item and \
                                             isinstance(langgraph_event_item["values"], dict) and \
                                             "messages" in langgraph_event_item["values"] and \
                                             isinstance(langgraph_event_item["values"]["messages"], list):
                                            # This is likely a "values" event containing the full state
                                            is_final_messages_event = True # Treat similarly
                                            messages_list = langgraph_event_item["values"]["messages"]
                                            if messages_list:
                                                last_message = messages_list[-1]
                                                if isinstance(last_message, dict) and (last_message.get("type") == "ai" or last_message.get("role") == "assistant"):
                                                    if not has_sent_content_for_turn:
                                                        item_content_chunk = last_message.get("content")
                                                    # Finish reason might be elsewhere or inferred
                                        
                                        elif langgraph_event_item.get("event") == "error" or langgraph_event_item.get("type") == "error":
                                            error_data = langgraph_event_item.get("data", {}); error_message = error_data.get("message", str(error_data)) if isinstance(error_data, dict) else str(error_data)
                                            item_content_chunk = f"Error from assistant: {error_message}"; item_finish_reason = "error"
                                        
                                        elif langgraph_event_item.get("event") == "on_graph_finish":
                                            if not current_turn_finish_reason: item_finish_reason = "stop" # Only if not already set by a message
                                        
                                        elif "run_id" in langgraph_event_item and ("thread_id" in langgraph_event_item or "attempt" in langgraph_event_item or "event" in langgraph_event_item): 
                                            print(f"DEBUG: Ignoring metadata/event: {json.dumps(langgraph_event_item)}"); continue 
                                        else: 
                                            print(f"WARNING: Unrecognized LangGraph stream event item (mode: {langgraph_stream_modes}): {json.dumps(langgraph_event_item)}"); continue 
                                        
                                        current_delta = ChatCompletionStreamChoiceDelta()
                                        
                                        if item_content_chunk is not None:
                                            if not has_sent_content_for_turn: # First content chunk for this AI turn
                                                current_delta.role = "assistant"
                                                has_sent_content_for_turn = True
                                            current_delta.content = str(item_content_chunk)
                                        
                                        if item_finish_reason and not current_turn_finish_reason: # Capture the first finish_reason for the turn
                                            current_turn_finish_reason = item_finish_reason
                                        
                                        # Yield chunk if it has content, establishes role, or has a finish reason
                                        if current_delta.role or current_delta.content or item_finish_reason:
                                            # If this event item provides a finish reason, use it for this chunk
                                            # Otherwise, if we already have a finish reason for the turn, don't send it yet
                                            # unless this is the final "messages" event.
                                            chunk_finish_reason = item_finish_reason
                                            
                                            # If it's a final messages event and we have a turn finish reason, ensure it's sent
                                            if is_final_messages_event and current_turn_finish_reason and not item_finish_reason:
                                                chunk_finish_reason = current_turn_finish_reason

                                            stream_choice = ChatCompletionStreamChoice(index=0, delta=current_delta, finish_reason=chunk_finish_reason)
                                            openai_stream_response = ChatCompletionStreamResponse(id=completion_id, created=request_created_time, model=model_display_name, choices=[stream_choice])
                                            json_to_yield = openai_stream_response.model_dump_json(exclude_none=True)
                                            print(f"DEBUG: Yielding OpenAI chunk (Threaded): {json_to_yield}") 
                                            yield f"data: {json_to_yield}\n\n"
                                        
                                        if item_finish_reason: 
                                            print(f"DEBUG: Finish reason '{item_finish_reason}' processed from an event item.")
                                            # If a definitive finish reason is received from any item,
                                            # it often means this logical AI response is complete.
                                            # The outer loop will send [DONE] after LangGraph closes the stream.

                                except json.JSONDecodeError: print(f"WARNING: JSONDecodeError on line: {line}")
                                except Exception as e: import traceback; print(f"ERROR processing line: {e} (line: {line})\n{traceback.format_exc()}"); final_error_delta = ChatCompletionStreamChoiceDelta(role="assistant", content=f"Proxy error: {str(e)}"); yield f"data: {ChatCompletionStreamResponse(id=completion_id, created=request_created_time, model=model_display_name, choices=[ChatCompletionStreamChoice(index=0, delta=final_error_delta, finish_reason='error')]).model_dump_json(exclude_none=True)}\n\n"; break 
                    print("DEBUG: LangGraph stream finished. Sending [DONE].")
                    yield "data: [DONE]\n\n"
                except httpx.HTTPStatusError as e_stream_connect: 
                    if e_stream_connect.response.status_code == 404 and attempt == 0:
                        print(f"INFO: Stream attempt 1 failed with 404 for thread '{thread_id}'. Will retry with a new thread.")
                        non_yielding_error = e_stream_connect 
                    else: 
                        error_message = f"LangGraph service error during stream connection: {e_stream_connect.response.status_code}"
                        try: error_message = e_stream_connect.response.json().get("detail", error_message)
                        except: pass
                        print(f"ERROR: {error_message}")
                        final_error_delta = ChatCompletionStreamChoiceDelta(role="assistant", content=error_message)
                        error_choice = ChatCompletionStreamChoice(index=0, delta=final_error_delta, finish_reason="error")
                        yield f"data: {ChatCompletionStreamResponse(id=completion_id,created=request_created_time,model=model_display_name, choices=[error_choice]).model_dump_json(exclude_none=True)}\n\n"
                        yield "data: [DONE]\n\n"
                except Exception as e_outer: 
                    import traceback; print(f"ERROR in stream_generator (Threaded): {e_outer}\n{traceback.format_exc()}"); yield f"data: {ChatCompletionStreamResponse(id=completion_id, created=request_created_time, model=model_display_name, choices=[ChatCompletionStreamChoice(index=0, delta=ChatCompletionStreamChoiceDelta(role='assistant', content=f'Proxy error: {e_outer}'), finish_reason='error')]).model_dump_json(exclude_none=True)}\n\n"; yield "data: [DONE]\n\n"
                if non_yielding_error: raise non_yielding_error
            try:
                return StreamingResponse(stream_generator(), media_type="text/event-stream")
            except httpx.HTTPStatusError as e: 
                if e.response.status_code == 404 and attempt == 0:
                    print(f"INFO: Stream attempt 1 resulted in 404. Clearing cached thread and retrying.")
                    if assistant_id_uuid in ACTIVE_THREADS_PER_MODEL: del ACTIVE_THREADS_PER_MODEL[assistant_id_uuid]
                    continue 
                else: 
                    detail = f"LangGraph API error: {e.response.status_code}"; 
                    try: detail = e.response.json().get("detail", detail)
                    except: pass
                    raise HTTPException(status_code=e.response.status_code, detail=detail)
            break 
        else: 
            langgraph_wait_url = f"{LANGGRAPH_API_BASE_URL}/threads/{thread_id}/runs/wait"
            try:
                print(f"DEBUG: Sending non-streaming (wait) request to LangGraph (Threaded): {langgraph_wait_url} with payload: {json.dumps(langgraph_payload)}")
                response = await async_client.post(langgraph_wait_url, json=langgraph_payload) 
                response.raise_for_status()
                langgraph_response_data = response.json() 
                print(f"DEBUG: Received non-streaming response from LangGraph /runs/wait (Threaded): {json.dumps(langgraph_response_data)}")
                assistant_reply_content = "Error: Could not parse response from /runs/wait."
                finish_reason = "error" 
                if isinstance(langgraph_response_data, dict):
                    if "messages" in langgraph_response_data and isinstance(langgraph_response_data["messages"], list):
                        if langgraph_response_data["messages"]:
                            last_message = langgraph_response_data["messages"][-1]
                            if isinstance(last_message, dict) and (last_message.get("type") == "ai" or last_message.get("role") == "assistant"):
                                assistant_reply_content = last_message.get("content", assistant_reply_content)
                                response_metadata = last_message.get("response_metadata")
                                if isinstance(response_metadata, dict):
                                    fr_meta = response_metadata.get("finish_reason")
                                    if fr_meta: finish_reason = fr_meta.lower() if fr_meta.upper() not in ["STOP", "LENGTH"] else fr_meta.upper()
                                elif not assistant_reply_content.startswith("Error:"): finish_reason = "stop" 
                    elif "output" in langgraph_response_data: 
                        lg_output = langgraph_response_data.get("output");
                        if isinstance(lg_output, dict): assistant_reply_content = lg_output.get("response", lg_output.get("content", assistant_reply_content))
                        elif isinstance(lg_output, str): assistant_reply_content = lg_output
                        if "finish_reason" in langgraph_response_data: finish_reason = langgraph_response_data["finish_reason"]
                        elif not assistant_reply_content.startswith("Error:"): finish_reason = "stop"
                    elif "response" in langgraph_response_data: assistant_reply_content = langgraph_response_data["response"];
                    if not assistant_reply_content.startswith("Error:") and finish_reason == "error": finish_reason = "stop"
                if not isinstance(assistant_reply_content, str): assistant_reply_content = json.dumps(assistant_reply_content)
                choice = ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=assistant_reply_content), finish_reason=finish_reason)
                openai_response = ChatCompletionResponse(id=completion_id, created=request_created_time, model=model_display_name, choices=[choice])
                print(f"DEBUG: Sending OpenAI non-streaming response (Threaded): {openai_response.model_dump_json(exclude_none=True)}")
                return openai_response
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404 and attempt == 0:
                    print(f"INFO: Non-streaming attempt 1 failed with 404 for thread '{thread_id}'. Will retry with a new thread.")
                    if assistant_id_uuid in ACTIVE_THREADS_PER_MODEL: del ACTIVE_THREADS_PER_MODEL[assistant_id_uuid]
                    continue 
                else: 
                    detail = f"Error from LangGraph /runs/wait (Threaded): {e.response.status_code}"; error_json = None
                    try: error_json = e.response.json(); detail = error_json.get("detail", str(error_json))
                    except: detail += f" - {e.response.text}"
                    print(f"ERROR: HTTPStatusError (Threaded /runs/wait): {detail}")
                    if e.response.status_code == 429: detail = "AI model overloaded (Backend 429)."
                    raise HTTPException(status_code=e.response.status_code, detail=detail)
            except Exception as e: 
                import traceback; print(f"ERROR: Internal error (Threaded /runs/wait): {str(e)}\n{traceback.format_exc()}"); 
                raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
            break 
    raise HTTPException(status_code=500, detail="Failed to complete chat operation after retries.")