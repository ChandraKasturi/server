import json
import typing
import datetime
from bson.objectid import ObjectId
from fastapi.responses import JSONResponse
from starlette import background

class UGJSONResponse(JSONResponse):
    """Custom JSON response class that properly handles MongoDB ObjectId and datetime objects."""
    
    media_type = "application/json"

    def __init__(
        self,
        content: typing.Any,
        status_code: int = 200,
        headers: typing.Mapping[str, str] | None = None,
        media_type: str | None = None,
        background: background.BackgroundTask | None = None,
    ) -> None:
        super().__init__(content, status_code, headers, media_type, background)

    def render(self, content: typing.Any) -> bytes:
        """Custom render method that converts ObjectId and datetime to strings."""
        def change_mongo_types(content):
            """Recursively convert MongoDB types to JSON-serializable types."""
            if isinstance(content, dict):
                for k, v in content.items():
                    if isinstance(v, dict):
                        change_mongo_types(content[k])
                    elif isinstance(v, list):
                        for i, item in enumerate(v):
                            change_mongo_types(item)
                    elif isinstance(v, ObjectId):
                        content[k] = str(v)
                    elif isinstance(v, datetime.datetime):
                        content[k] = v.isoformat()
            elif isinstance(content, list):
                for item in content:
                    change_mongo_types(item)
            
            return content
        
        # Convert MongoDB types and serialize to JSON
        processed_content = change_mongo_types(content)
        
        return json.dumps(
            processed_content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8") 