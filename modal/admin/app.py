import os
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional

import requests  # type: ignore
import sentry_sdk
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator

import modal
import supabase
from phosphobot.am.act import ACT, ACTSpawnConfig
from phosphobot.am.base import TrainingRequest
from phosphobot.am.gr00t import Gr00tN1, Gr00tSpawnConfig
from phosphobot.models import CancelTrainingRequest

phosphobot_dir = (
    Path(__file__).parent.parent.parent.parent / "phosphobot" / "phosphobot"
)
admin_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "loguru",
        "supabase",
        "fastapi[standard]",
        "pydantic==2.10.6",
        "requests",
        "httpx>=0.28.1",
        "pydantic>=2.10.5",
        "fastparquet>=2024.11.0",
        "ffmpeg-python>=0.2.0",
        "loguru>=0.7.3",
        "numpy<2",
        "opencv-python-headless>=4.0",
        "rich>=13.9.4",
        "pandas-stubs>=2.2.2.240807",
        "huggingface-hub>=0.29.0",
        "json-numpy>=2.1.0",
        "fastapi>=0.115.11",
        "zmq>=0.0.0",
        "av>=14.2.1",
        "sentry-sdk",
    )
    .pip_install_from_pyproject(
        pyproject_toml=str(phosphobot_dir / "pyproject.toml"),
    )
    # .add_local_dir(
    #     local_path=phosphobot_dir,
    #     remote_path="/root/phosphobot",
    #     # ignore if .venv is in path
    #     ignore=lambda p: ".venv" in str(p),
    # )
    .add_local_python_source("phosphobot")
)

# TODO: add HF_TRANSFER for faster downloads?
# Used for web endpoints
auth_scheme = HTTPBearer()

MINUTES = 60  # seconds
BASE_TRAININGS_LIMIT = 2
WHITELISTED_TRAININGS_LIMIT = 8
PRO_TRAININGS_LIMIT = 8

app = modal.App("admin-api")

# Get the serving functions by name
serve_anywhere = modal.Function.from_name("gr00t-server", "serve_anywhere")
serve_us_east = modal.Function.from_name("gr00t-server", "serve_us_east")
serve_us_west = modal.Function.from_name("gr00t-server", "serve_us_west")
serve_eu = modal.Function.from_name("gr00t-server", "serve_eu")
serve_ap = modal.Function.from_name("gr00t-server", "serve_ap")
serve_act = modal.Function.from_name("act-server", "serve")
serve_smolvla = modal.Function.from_name("smolvla-server", "serve")
# Get the training functions by name
train_gr00t = modal.Function.from_name("gr00t-server", "train")
train_act = modal.Function.from_name("act-server", "train")
# Paligemma warmup function
paligemma_warmup = modal.Function.from_name("paligemma-detector", "warmup_model")

if os.getenv("MODAL_ENVIRONMENT") == "production":
    sentry_sdk.init(
        dsn="https://afa38885e368d772d8eced1bce325604@o4506399435325440.ingest.us.sentry.io/4509203019005952",
        traces_sample_rate=1.0,
        environment="production",
    )

# Build the zone to function mapping
ZONE_TO_FUNCTION_GR00T = {
    "anywhere": serve_anywhere,
    "us-east": serve_us_east,
    "us-west": serve_us_west,
    "eu": serve_eu,
    "ap": serve_ap,
}

ZONE_TO_FUNCTION_ACT = {
    "anywhere": serve_act,
    "us-east": serve_act,
    "us-west": serve_act,
    "eu": serve_act,
    "ap": serve_act,
}

ZONE_TO_FUNCTION_SMOLVLA = {
    "anywhere": serve_smolvla,
    "us-east":  serve_smolvla,
    "us-west":  serve_smolvla,
    "eu":       serve_smolvla,
    "ap":       serve_smolvla,
}


# Model mapping
MODEL_TO_ZONE = {
    "gr00t": ZONE_TO_FUNCTION_GR00T,
    "ACT": ZONE_TO_FUNCTION_ACT,
    "ACT_BBOX": ZONE_TO_FUNCTION_ACT,  # ACT_BBOX uses the same serving function as ACT
    "smolvla": ZONE_TO_FUNCTION_SMOLVLA,    
}

# Mapping from countryCode to best region
COUNTRY_TO_REGION = {
    # North America
    # Default to us-east for US (will be refined at next step with longitude)
    "US": "us-east",
    "CA": "us-west",
    "MX": "us-west",
    # Europe
    "GB": "eu",
    "DE": "eu",
    "FR": "eu",
    "IT": "eu",
    "ES": "eu",
    "NL": "eu",
    "BE": "eu",
    "SE": "eu",
    "PL": "eu",
    "AT": "eu",
    "CH": "eu",
    "DK": "eu",
    "FI": "eu",
    "NO": "eu",
    "IE": "eu",
    "PT": "eu",
    "GR": "eu",
    "CZ": "eu",
    # Asia-Pacific
    "JP": "ap",
    "CN": "ap",
    "KR": "ap",
    "IN": "ap",
    "AU": "ap",
    "NZ": "ap",
    "SG": "ap",
    "HK": "ap",
    "MY": "ap",
    "TH": "ap",
    "ID": "ap",
    "PH": "ap",
    "VN": "ap",
    "TW": "ap",
}


class IpLocationInfo(BaseModel):
    status: str
    countryCode: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


def get_location_from_ip(ip_address=None) -> Optional[IpLocationInfo]:
    """
    Get geolocation data from IP address using ip-api.com
    Returns the JSON response from the API or None if the request fails
    """
    # Construct URL with IP if provided, otherwise use current IP
    url = "http://ip-api.com/json/"
    if ip_address:
        url += ip_address

    # Add fields parameter to request only what we need
    url += "?fields=status,message,countryCode,lat,lon"

    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            location_data = IpLocationInfo.model_validate(data)
            if location_data.status == "success":
                return location_data
            else:
                logger.error(f"IP geolocation failed: {data.get('message')}")
        else:
            logger.error(
                f"IP geolocation request failed with status code: {response.status_code}"
            )
    except Exception as e:
        logger.error(f"Error fetching IP geolocation: {e}")

    return None


def determine_best_region(ip_address=None):
    """
    Determine the best region for serving based on the client's IP address
    Returns one of: "us-east", "us-west", "eu", "ap", or "anywhere" (default fallback)
    """
    location_data = get_location_from_ip(ip_address)
    if not location_data:
        logger.warning("Could not determine location from IP, defaulting to 'anywhere'")
        return "anywhere"

    country_code = location_data.countryCode
    region = "anywhere"  # Default fallback region

    # If country code exists in our mapping, use that region
    if country_code in COUNTRY_TO_REGION:
        region = COUNTRY_TO_REGION[country_code]

        # Special case for US - determine east vs west based on longitude
        if country_code == "US" and location_data.lon is not None:
            # Roughly split US at longitude -100
            if location_data.lon < -100:
                region = "us-west"
            else:
                region = "us-east"

        logger.info(f"Determined region {region} from country code {country_code}")
    else:
        logger.info(
            f"Country code {country_code} not found in mapping, defaulting to 'anywhere'"
        )

    # HOTFIX: if region == "ap", switch to "anywhere"
    if region == "ap":
        logger.info(
            f"Country {country_code} is AP, defaulting to 'anywhere' due to issues with modal serving in AP"
        )
        region = "anywhere"

    logger.debug(f"Determined region: {region} for IP {ip_address}")
    return region


def generate_huggingface_model_name(dataset):
    # Generate 10 random alphanumeric characters (similar to the JS version)
    random_chars = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
    return f"{dataset}-{random_chars}"


class StartServerRequest(BaseModel):
    model_id: str
    model_type: Literal["gr00t", "ACT", "ACT_BBOX"]
    timeout: Annotated[int, Field(default=15 * MINUTES, ge=0)]
    region: Optional[Literal["us-east", "us-west", "eu", "ap", "anywhere"]] = None
    model_specifics: Gr00tSpawnConfig | ACTSpawnConfig

    @field_validator("timeout", mode="before")
    def clamp_timeout(cls, v: int) -> int:
        return max(0, min(v, 60 * MINUTES))

    @model_validator(mode="after")
    def validate_model_specifics(self) -> "StartServerRequest":
        correspondance = {
            "gr00t": Gr00tSpawnConfig,
            "ACT": ACTSpawnConfig,
            "ACT_BBOX": ACTSpawnConfig,
        }

        # Make sure the model_specifics is of the right type
        if not isinstance(self.model_specifics, correspondance[self.model_type]):
            raise ValueError(
                f"model_specifics should be of type {correspondance[self.model_type]}"
            )
        return self


class ServerInfo(BaseModel):
    server_id: int
    url: str
    port: int
    tcp_socket: tuple[str, int]
    model_id: str
    timeout: int


class SupabaseServersTable(BaseModel):
    status: Literal["running", "stopped"]
    host: str | None = None
    port: int | None = None
    user_id: str
    model_id: str
    model_type: Literal["gr00t", "ACT", "ACT_BBOX"]
    timeout: int | None = None
    requested_at: Optional[str] = None
    terminated_at: Optional[str] = None
    region: Optional[str] = None


class ModelInfo(BaseModel):
    """
    Publicly available model info
    """

    status: Literal["succeeded", "failed", "running"]
    dataset_name: str
    model_name: Optional[str] = None
    requested_at: datetime
    terminated_at: Optional[datetime] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    used_wandb: Optional[bool] = None
    logs: Optional[str] = None
    train_test_split: Optional[float] = None
    model_type: Optional[str] = None
    training_params: Dict[str, Any] = Field(default_factory=dict)
    # Config will be used by /spawn in Gr00tSpawnConfig | ACTSpawnConfig
    config: Optional[Gr00tSpawnConfig | ACTSpawnConfig] = None


class ModelInfoResponse(BaseModel):
    total_count: int
    model_infos: List[ModelInfo]


class ModelStatusResponse(BaseModel):
    model_url: str
    model_status: Literal["running", "succeeded", "failed", "not-found"]
    model_info: Optional[ModelInfo] = None


class PublicUser(BaseModel):
    """
    User info that can be sent to the client
    Not all fields can be sent to the client
    """

    id: str
    plan: Literal["pro"] | None = None


@app.function(
    image=admin_image,
    allow_concurrent_inputs=1000,
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("supabase")],
    # We keep at least one instance of the app running
    min_containers=1,
)
@modal.asgi_app()
def fastapi_app():
    from datetime import datetime, timezone

    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    web_app = FastAPI()
    supabase_client = supabase.Client(
        supabase_url=os.environ["SUPABASE_URL"],
        supabase_key=os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.warning(f"HTTPException: {exc.status_code} - {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    # Add an health route
    @web_app.get("/health")
    async def get_health():
        """
        Health check endpoint
        """
        return {"status": "ok", "url": await fastapi_app.web_url}

    @web_app.get("/models", response_model=ModelInfoResponse)
    async def get_models():
        """
        Get all models in database with a status of "succeeded"
        """
        # To do so, we want to list all distinct values of model_name where status is "succeeded", ordered by descending created_at
        model_infos = (
            supabase_client.table("trainings")
            .select("*")
            .eq("status", "succeeded")
            .order("requested_at", desc=True)
            .limit(10000)
            .execute()
        )

        response = []

        # Log the number of model_infos
        logger.info(f"Found {len(model_infos.data)} model_infos")

        # Validate the model_infos, ignore the ones that are not valid
        for model_info in model_infos.data:
            try:
                ModelInfo.model_validate(model_info)
                response.append(model_info)
            except Exception:
                # Do nothing
                pass

        logger.info(f"Found {len(response)} valid model_infos")

        return ModelInfoResponse(model_infos=response, total_count=len(response))

    @web_app.get("/models/{username}/{model_id}")
    async def get_model(username: str, model_id: str):
        """
        Get a model from the database
        """
        model_url = f"https://huggingface.co/{username}/{model_id}"
        # Query HF, same approach than in the local teleop backend
        response = requests.get(model_url)
        response_text = response.text.lower()
        if any(keyword in response_text for keyword in ["error traceback"]):
            return ModelStatusResponse(model_url=model_url, model_status="failed")
        elif any(
            keyword in response_text
            for keyword in ["epochs", "batch size", "training steps"]
        ):
            # Get more info by queying supabase
            # We take the first row of the trainings table where model_name == model_id
            model_info = (
                supabase_client.table("trainings")
                .select("*")
                .eq("model_name", f"{username}/{model_id}")
                .limit(1)
                .execute()
            )
            if model_info.data:
                row = model_info.data[0]

                # This will automatically handle None from row for any Optional fields
                info = ModelInfo.model_validate(row)

                # We fetch the configs (need to spwan a server)
                model_types: Dict[str, type[ACT | Gr00tN1]] = {
                    "gr00t": Gr00tN1,
                    "ACT": ACT,
                }
                model_used = model_types[str(info.model_type)]

                info.config = model_used.fetch_spawn_config(
                    model_id=f"{username}/{model_id}"
                )

                return ModelStatusResponse(
                    model_url=model_url,
                    model_status="succeeded",
                    model_info=info,
                )
            else:
                return ModelStatusResponse(
                    model_url=model_url, model_status="not-found"
                )
        elif response.status_code == 200:
            return ModelStatusResponse(model_url=model_url, model_status="running")
        else:
            return ModelStatusResponse(model_url=model_url, model_status="not-found")

    @web_app.post("/spawn")
    async def spawn_server_for_model(
        raw_request: Request,
        request: StartServerRequest,
        token: HTTPAuthorizationCredentials = Depends(auth_scheme),
    ):
        """
        POST to this endpoint to start a gr00t inference server
        """
        # See https://modal.com/docs/guide/webhooks#token-based-authentication for token-based auth

        try:
            user = supabase_client.auth.get_user(jwt=token.credentials)
            logger.debug(f"User: {user.user.email} ({user.user.id}) spawning server")
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        active_servers = (
            supabase_client.table("servers")
            .select("*")
            .eq("user_id", user.user.id)
            .eq("status", "running")
            .eq("model_id", request.model_id)
            .is_("terminated_at", "null")
            .limit(1)
            .execute()
        )

        if active_servers.data:
            # Return the existing server info into a ServerInfo object
            # if it's the same model_id
            row = active_servers.data[0]

            if row["status"] == "running" and row["region"] is None:
                # If it has been more than 5 minutes, we can assume there was an error
                # and we can restart the server
                if (
                    datetime.now(timezone.utc)
                    - datetime.fromisoformat(row["requested_at"])
                ).total_seconds() > 5 * MINUTES:
                    # Restart the server
                    supabase_client.table("servers").update(
                        {
                            "status": "stopped",
                            "terminated_at": datetime.now(timezone.utc).isoformat(),
                        }
                    ).eq("id", row["id"]).execute()
                    # Kill the modal function
                    # TODO: we can't do that because the modal_function_call_id is not stored.
                    # try:
                    #     modal.FunctionCall.from_id(
                    #         row["modal_function_call_id"]
                    #     ).cancel()
                    # except Exception as e:
                    #     logger.error(f"Error cancelling modal function: {e}")
                    #     # We can ignore this error, the server will be restarted anyway

                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="There has been an error with the server. Please try again and reach out on Discord if it persists.",
                    )

                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Server is warming up... please wait and try again",
                )
            logger.info(
                f"User {user.user.email} already has an active server for model {request.model_id}"
            )
            server_info = ServerInfo(
                server_id=row["id"],
                url=row["url"],
                port=row["port"],  # This is the port on which we start ACT
                tcp_socket=(str(row["host"]), int(row["tcp_port"])),
                model_id=row["model_id"],
                timeout=row["timeout"],
            )
            return server_info

        server_data = SupabaseServersTable(
            status="running",
            user_id=user.user.id,
            model_id=request.model_id,
            model_type=request.model_type,
            timeout=request.timeout,
            region=request.region,
        )
        row = (
            supabase_client.table("servers")
            .insert(server_data.model_dump(exclude_unset=True))
            .execute()
        )
        server_id = row.data[0]["id"]

        # Determine region from IP if not specified
        if request.region is None:
            # Get client IP address - check X-Forwarded-For header first
            client_ip = None
            forwarded_for = raw_request.headers.get("X-Forwarded-For")
            if forwarded_for:
                # X-Forwarded-For can contain multiple IPs, the first one is the client
                client_ip = forwarded_for.split(",")[0].strip()
                logger.debug(f"Using IP from X-Forwarded-For header: {client_ip}")
            elif raw_request and hasattr(raw_request, "client") and raw_request.client:
                client_ip = raw_request.client.host
                logger.debug(f"Using direct client IP: {client_ip}")

            # Determine best region based on IP
            request.region = determine_best_region(client_ip)
            logger.info(f"Determined region {request.region} for IP {client_ip}")

        # Default to "anywhere" if no region was determined
        if request.region is None:
            request.region = "anywhere"

        # Start server
        with modal.Queue.ephemeral() as q:
            # Get the function to serve from the zone to function mapping
            serve = MODEL_TO_ZONE[request.model_type][request.region]
            # Spawn the serve function with the queue
            serve.spawn(
                model_id=request.model_id,
                server_id=server_id,
                timeout=request.timeout,
                model_specifics=request.model_specifics,
                q=q,
            )

            if request.model_type == "ACT_BBOX":
                paligemma_warmup.spawn()

            # Get the tunnel information from the queue
            result = q.get()

        server_info = ServerInfo.model_validate(result)

        try:
            supabase_data = {
                "url": server_info.url,
                "port": server_info.port,
                "host": server_info.tcp_socket[0],
                "tcp_port": server_info.tcp_socket[1],
                "region": request.region,
                "model_id": request.model_id,
            }
            supabase_client.table("servers").update(supabase_data).eq(
                "id", server_id
            ).execute()
        except Exception as e:
            logger.error(f"Error inserting server data into database: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error inserting server data into database",
            )

        logger.success(f"Server started: {server_info.model_dump()}")
        return server_info

    @web_app.post("/train")
    async def start_training(
        request: TrainingRequest,
        token: HTTPAuthorizationCredentials = Depends(auth_scheme),
    ):
        # TODO: factorize using dependency injection
        try:
            user = supabase_client.auth.get_user(jwt=token.credentials)
            logger.debug(f"User: {user.user.email} ({user.user.id}) spawning server")
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        active_trainings = (
            supabase_client.table("trainings")
            .select("*")
            .eq("user_id", user.user.id)
            .eq("status", "running")
            .is_("terminated_at", "null")
            .limit(10)
            .execute()
        )

        id_whitelist = [
            "ab9b958d-ca0b-4d83-862f-60ba4ed35398",
            "a9cff082-9c44-4bcb-b262-0edc31c067c0",
        ]
        user_id = user.user.id

        logger.info(f"Active trainings: {active_trainings.data}")

        # Check if the user is a pro user
        user_data = (
            supabase_client.table("users").select("*").eq("id", user.user.id).execute()
        )
        user_plan: Literal["pro"] | None = None
        if user_data.data:
            user_plan = user_data.data[0].get("plan", None)

        # Handle timeout based on user plan. Default to 3 hours for normal users.
        timeout_seconds = 3 * 60 * 60  # 3 hours in seconds
        if user_plan == "pro" or user_id in id_whitelist:
            # Pro users or whitelisted users get a longer timeout
            logger.info(
                f"User {user_id} is a PRO user or whitelisted, extending timeout to 12 hours"
            )
            timeout_seconds = 12 * 60 * 60  # 12 hours in seconds

        if user_id in id_whitelist:
            logger.info(f"User {user_id} is launching a training")
            if len(active_trainings.data) >= WHITELISTED_TRAININGS_LIMIT:
                logger.warning(
                    f"User {user_id} is whitelisted but already has 8 active trainings"
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="You have too many active trainings, please wait for one to finish",
                )
        else:
            # Normal flow
            if user_plan is None and len(active_trainings.data) >= BASE_TRAININGS_LIMIT:
                logger.warning(f"User {user_id} already has an active training")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"You have already {BASE_TRAININGS_LIMIT} active trainings, please wait for one to finish or upgrade to a PRO plan",
                )
            elif (
                user_plan == "pro" and len(active_trainings.data) >= PRO_TRAININGS_LIMIT
            ):
                logger.warning(f"User {user_id} already has an active training")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"You have already {PRO_TRAININGS_LIMIT} active trainings, please wait for it to finish or upgrade to a PRO plan",
                )

        supabase_data = {
            "status": "running",
            "user_id": user.user.id,
            "used_wandb": request.wandb_api_key is not None,
        }

        request_data = request.model_dump(
            exclude_unset=True, exclude={"wandb_api_key", "training_params"}
        )
        if request.training_params:
            supabase_data["training_params"] = request.training_params.model_dump()
        for key, value in request_data.items():
            supabase_data[key] = value

        try:
            row = supabase_client.table("trainings").insert(supabase_data).execute()
            training_id = row.data[0]["id"]
        except Exception as e:
            logger.error(f"Error inserting training data into database: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error inserting training data into database",
            )

        models_dict = {
            "gr00t": train_gr00t,
            "ACT": train_act,
            "ACT_BBOX": train_act,
        }

        logger.info(f"Starting training for {request.model_type}")
        logger.info(f"Training params: {request.training_params}")

        # We pass all the parameters to the training function
        spawn_response = models_dict[request.model_type].spawn(
            **request.model_dump(exclude={"training_params"}),
            training_id=training_id,
            training_params=request.training_params,
            timeout_seconds=timeout_seconds,
        )
        # Update the training row with the function_call_id
        try:
            supabase_client.table("trainings").update(
                {"modal_function_call_id": spawn_response.object_id}
            ).eq("id", training_id).execute()
        except Exception as e:
            logger.error(f"Error updating training data in database: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error updating training data in database",
            )

        return {"status": "ok", "training_id": training_id}

    @web_app.post("/cancel")
    async def cancel_training(
        request: CancelTrainingRequest,
        token: HTTPAuthorizationCredentials = Depends(auth_scheme),
    ):
        """
        Cancel a training job by ID.
        """
        try:
            user = supabase_client.auth.get_user(jwt=token.credentials)
            logger.debug(
                f"User: {user.user.email} ({user.user.id}) cancelling training"
            )
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            raise HTTPException(
                status_code=401,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check if the training exists and belongs to the user
        training = (
            supabase_client.table("trainings")
            .select("*")
            .eq("id", request.training_id)
            .eq("user_id", user.user.id)
            .is_("terminated_at", "null")
            .execute()
        )

        if not training.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Training not found or does not belong to the user",
            )

        status = "ok"
        message = f"Training {request.training_id} canceled successfully"

        try:
            # Cancel the training job in Modal
            modal_function_call_id = training.data[0].get("modal_function_call_id")
            if not modal_function_call_id:
                raise ValueError(
                    f"No modal_function_call_id found for training {request.training_id}"
                )
            modal.FunctionCall.from_id(modal_function_call_id).cancel()
        except Exception as e:
            logger.error(f"Error cancelling training: {e}")
            status = "error"
            message = f"Error cancelling training: {e}"

        # Update the training status to canceled anyways
        supabase_client.table("trainings").update(
            {
                "status": "canceled",
                "terminated_at": datetime.now(timezone.utc).isoformat(),
            }
        ).eq("id", request.training_id).execute()

        return {"status": status, "message": message}

    @web_app.get("/users/me", response_model=PublicUser)
    async def get_user(
        token: HTTPAuthorizationCredentials = Depends(auth_scheme),
    ) -> PublicUser:
        """
        Get the user info from the supabase public.users table
        """
        user = supabase_client.auth.get_user(jwt=token.credentials)

        # Query the public.users table
        user_data = (
            supabase_client.table("users").select("*").eq("id", user.user.id).execute()
        )

        # Can be empty
        if not user_data.data:
            public_user_data = PublicUser(id=user.user.id, plan=None)
        else:
            public_user_data = PublicUser.model_validate(user_data.data[0])

        return public_user_data

    # Required by modal
    return web_app
