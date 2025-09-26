import os
import random
import string
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional

import modal
import requests  # type: ignore
import sentry_sdk
from fastapi import Depends, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import supabase
from phosphobot.am.act import ACT, ACTSpawnConfig
from phosphobot.am.base import TrainingRequest
from phosphobot.am.gr00t import Gr00tN1, Gr00tSpawnConfig
from phosphobot.am.pi05 import Pi05, Pi05SpawnConfig
from phosphobot.am.smolvla import SmolVLA, SmolVLASpawnConfig
from phosphobot.models import CancelTrainingRequest, ChatRequest, ChatResponse

phosphobot_dir = (
    Path(__file__).parent.parent.parent.parent.parent / "phosphobot" / "phosphobot"
)
admin_image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install(
        "ffmpeg-python>=0.2.0",
        "stripe",
        "httpx",
        "google-genai>=1.35.0",
    )
    .pip_install_from_pyproject(
        pyproject_toml=str(phosphobot_dir / "pyproject.toml"),
    )
    .add_local_python_source("phosphobot")
)

# TODO: add HF_TRANSFER for faster downloads?
# Used for web endpoints
auth_scheme = HTTPBearer()

MINUTES = 60  # seconds
BASE_TRAININGS_LIMIT = 2
WHITELISTED_TRAININGS_LIMIT = 8
PRO_TRAININGS_LIMIT = 8
BASE_MONTHLY_LIMIT = 5
PRO_MONTHLY_LIMIT = 100
BASE_TIMEOUT = 1 * 60 * MINUTES  # 1 hour in seconds
PRO_TIMEOUT = 3 * 60 * MINUTES  # 3 hours in seconds
ALLOWLISTED_TIMEOUT = 12 * 60 * MINUTES  # 12 hours in seconds
# Max allowed time for a server to cold start before we assume it failed
TIMEOUT_SERVER_NOT_STARTED = 3 * MINUTES

app = modal.App("admin-api")

# Get the serving functions by name
serve_anywhere = modal.Function.from_name("gr00t-server", "serve_anywhere")
serve_us_east = modal.Function.from_name("gr00t-server", "serve_us_east")
serve_us_west = modal.Function.from_name("gr00t-server", "serve_us_west")
serve_eu = modal.Function.from_name("gr00t-server", "serve_eu")
serve_ap = modal.Function.from_name("gr00t-server", "serve_ap")
serve_act = modal.Function.from_name("act-server", "serve")
serve_pi05 = modal.Function.from_name("pi0.5-server", "serve")
serve_smolvla = modal.Function.from_name("smolvla-server", "serve")
# Get the training functions by name
train_gr00t = modal.Function.from_name("gr00t-server", "train")
train_act = modal.Function.from_name("act-server", "train")
train_pi05 = modal.Function.from_name("pi0.5-server", "train")
train_smolvla = modal.Function.from_name("smolvla-server", "train")
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

ZONE_TO_FUNCTION_PI05 = {
    "anywhere": serve_pi05,
    "us-east": serve_pi05,
    "us-west": serve_pi05,
    "eu": serve_pi05,
    "ap": serve_pi05,
}

ZONE_TO_FUNCTION_SMOLVLA = {
    "anywhere": serve_smolvla,
    "us-east": serve_smolvla,
    "us-west": serve_smolvla,
    "eu": serve_smolvla,
    "ap": serve_smolvla,
}


# Model mapping
MODEL_TO_ZONE = {
    "gr00t": ZONE_TO_FUNCTION_GR00T,
    "ACT": ZONE_TO_FUNCTION_ACT,
    "ACT_BBOX": ZONE_TO_FUNCTION_ACT,  # ACT_BBOX uses the same serving function as ACT
    "pi0.5": ZONE_TO_FUNCTION_PI05,
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


def determine_best_region(
    ip_address=None,
) -> Literal["us-east", "us-west", "eu", "ap", "anywhere"]:
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


def generate_wandb_run_id(model_name: Optional[str] = None) -> str:
    """
    Generates a random run id for WandB, wtarting with an adjective and an animal (human readable)
    """
    adjectives = [
        "gentle",
        "happy",
        "aggressive",
        "proud",
        "sad",
        "funny",
        "serious",
        "boring",
        "exciting",
        "junior",
        "senior",
        "mega",
        "blurry",
        "digital",
        "bullish",
        "bearish",
        "fast",
        "massive",
        "tiny",
        "maxxed",
        "giga",
        "supa",
        "hot",
        "cold",
        "frozen",
        "burning",
        "cooked",
        "raw",
        "testing",
        "dark",
        "googoo",
        "gaga",
        "red",
        "nitro",
    ]
    animals = [
        "panda",
        "tiger",
        "lion",
        "elephant",
        "giraffe",
        "dog",
        "cat",
        "snake",
        "bird",
        "fish",
        "whale",
        "dolphin",
        "shark",
        "crocodile",
        "junior",
        "pl",
        "gazelle",
        "cheetah",
        "bear",
        "bull",
        "wolf",
        "fox",
        "horse",
        "pony",
        "chicken",
        "penguin",
        "tarentula",
        "scorpion",
        "spider",
        "ant",
        "bee",
        "rat",
        "doggy",
        "rabbit",
        "kitty",
        "quark",
        "muskox",
        "monster",
    ]

    # Choose a random adjective and an animal
    adjective = random.choice(adjectives)
    animal = random.choice(animals)

    # Add a random number between 1 and 1000
    number = random.randint(1, 1000)

    if not model_name:
        return f"{adjective}-{animal}-{number}"
    else:
        # If a model name is provided, append it to the run id
        # pop the user/ part of the model_name
        model_name = model_name.split("/")[-1]  # Remove user/ if present
        return f"{adjective}-{animal}-{model_name}"


class StartServerRequest(BaseModel):
    model_id: str
    model_type: Literal["gr00t", "ACT", "ACT_BBOX", "pi0.5", "smolvla"]
    timeout: Annotated[int, Field(default=15 * MINUTES, ge=0, le=15 * MINUTES)]
    region: Optional[Literal["us-east", "us-west", "eu", "ap", "anywhere"]] = None
    model_specifics: (
        Gr00tSpawnConfig | ACTSpawnConfig | Pi05SpawnConfig | SmolVLASpawnConfig
    )
    checkpoint: Optional[int] = None

    @field_validator("timeout", mode="before")
    def clamp_timeout(cls, v: int) -> int:
        return max(0, min(v, 60 * MINUTES))

    @model_validator(mode="after")
    def validate_model_specifics(self) -> "StartServerRequest":
        correspondance = {
            "gr00t": Gr00tSpawnConfig,
            "ACT": ACTSpawnConfig,
            "ACT_BBOX": ACTSpawnConfig,
            "pi0.5": Pi05SpawnConfig,
            "smolvla": SmolVLASpawnConfig,
        }

        # Make sure the model_specifics is of the right type
        if not isinstance(self.model_specifics, correspondance[self.model_type]):
            raise ValueError(
                f"model_specifics should be of type {correspondance[self.model_type]}"
            )
        return self


class ServerInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    server_id: int
    url: str
    port: int
    tcp_socket: tuple[str, int]
    model_id: str
    timeout: int
    modal_function_call_id: str


class SupabaseServersTable(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    status: Literal["requested", "running", "stopped", "failed"]
    host: str | None = None
    port: int | None = None
    user_id: str
    model_id: str
    model_type: Literal["gr00t", "ACT", "ACT_BBOX", "pi0.5", "smolvla"]
    timeout: int | None = None
    requested_at: Optional[str] = None
    started_at: Optional[str] = None
    terminated_at: Optional[str] = None
    region: Optional[str] = None
    checkpoint: Optional[int] = None
    tcp_port: Optional[int] = None
    url: Optional[str] = None
    modal_function_call_id: Optional[str] = None


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
    model_type: Optional[str] = None
    training_params: Dict[str, Any] = Field(default_factory=dict)
    config: Optional[
        Gr00tSpawnConfig | ACTSpawnConfig | Pi05SpawnConfig | SmolVLASpawnConfig
    ] = None


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
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("supabase"),
        modal.Secret.from_name("stripe"),
        modal.Secret.from_name("gemini"),
    ],
    # We keep at least one instance of the app running
    min_containers=1,
)
@modal.concurrent(max_inputs=1000)
@modal.asgi_app()
def fastapi_app():
    from datetime import datetime, timezone

    import stripe
    from fastapi import FastAPI, Request
    from fastapi.exceptions import HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from supabase_auth.types import User as SupabaseUser

    stripe.api_key = os.environ["STRIPE_API_KEY"]

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
        return {"status": "ok", "url": fastapi_app.web_url}

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
                model_types: Dict[str, type[ACT | Gr00tN1 | Pi05 | SmolVLA]] = {
                    "gr00t": Gr00tN1,
                    "ACT": ACT,
                    "pi0.5": Pi05,
                    "smolvla": SmolVLA,
                }
                model_used = model_types[str(info.model_type)]

                fetched_config = model_used.fetch_spawn_config(
                    model_id=f"{username}/{model_id}"
                )
                info.config = fetched_config

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

    async def _stop_servers_of_user(user_id: str):
        """
        Stop all the currently running servers for the user
        This is used by the auth middleware to stop servers when the user logs out

        Return: the list of server IDs that were stopped
        """
        logger.debug(f"Stopping servers for user {user_id}")
        active_servers = (
            supabase_client.table("servers")
            .select("*")
            .eq("user_id", user_id)
            .eq("status", "running")
            .execute()
        )

        if not active_servers.data:
            logger.info("No active servers to cancel")
            return []

        # Update the status of all active servers to "stopped"
        supabase_client.table("servers").update(
            {
                "status": "stopped",
                "terminated_at": datetime.now(timezone.utc).isoformat(),
            }
        ).in_("id", [server["id"] for server in active_servers.data]).execute()

        # Cancel the modal function calls for each server
        for server in active_servers.data:
            server_id = server["id"]
            try:
                logger.debug(
                    f"Cancelling Modal function {server['modal_function_call_id']} for server {server_id}"
                )
                modal_function = modal.FunctionCall.from_id(
                    server["modal_function_call_id"]
                )
                # Stop the container
                modal_function.cancel(terminate_containers=True)
            except Exception as e:
                logger.error(f"Error stopping server {server_id}: {e}")

        # Return the list of server IDs that were stopped
        return [server["id"] for server in active_servers.data]

    @web_app.post("/spawn")
    async def spawn_server_for_model(
        raw_request: Request,
        request: StartServerRequest,
        token: HTTPAuthorizationCredentials = Depends(auth_scheme),
    ):
        """
        POST to this endpoint to start an inference server
        """
        # See https://modal.com/docs/guide/webhooks#token-based-authentication for token-based auth

        user = supabase_client.auth.get_user(jwt=token.credentials)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        logger.debug(f"User: {user.user.email} ({user.user.id}) spawning server")
        # Stop any existing servers for the user
        await _stop_servers_of_user(user_id=user.user.id)

        new_server = (
            supabase_client.table("servers")
            .insert(
                {
                    "status": "requested",
                    "user_id": user.user.id,
                    "model_id": request.model_id,
                    "model_type": request.model_type,
                    "timeout": request.timeout,
                    "region": request.region,
                    "checkpoint": request.checkpoint,
                    "requested_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            .execute()
        )
        server_id = new_server.data[0]["id"]

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
            spawn_response = serve.spawn(
                model_id=request.model_id,
                checkpoint=request.checkpoint,
                server_id=server_id,
                timeout=request.timeout,
                model_specifics=request.model_specifics,
                q=q,
            )

            if request.model_type == "ACT_BBOX":
                paligemma_warmup.spawn()

            # Get the tunnel information from the queue
            result: dict | None = q.get()
            if result is None:
                logger.error("No tunnel info received from queue")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to start server, no tunnel info received",
                )

        server_info = ServerInfo(
            modal_function_call_id=spawn_response.object_id, **result
        )

        try:
            update_payload = {
                "url": server_info.url,
                "port": server_info.port,
                "host": server_info.tcp_socket[0],
                "tcp_port": server_info.tcp_socket[1],
                "region": request.region,
                "model_id": request.model_id,
                "status": "running",
                "modal_function_call_id": server_info.modal_function_call_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            supabase_client.table("servers").update(update_payload).eq(
                "id", server_id
            ).execute()
        except Exception as e:
            logger.error(f"Error inserting server data into database: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error inserting server data into database",
            )

        logger.success(f"Server started:\n{server_info.model_dump_json(indent=4)}")
        return server_info

    @web_app.post("/stop")
    async def stop_inference(
        token: HTTPAuthorizationCredentials = Depends(auth_scheme),
    ):
        """
        Stop all the currently running servers for the user
        """
        user = supabase_client.auth.get_user(jwt=token.credentials)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        logger.debug(f"User: {user.user.email} ({user.user.id}) cancelling servers")

        stopped_servers = await _stop_servers_of_user(user_id=user.user.id)
        if not stopped_servers:
            return {"detail": "No active servers to cancel"}

        return {"detail": "All active servers cancelled successfully"}

    @web_app.post("/train")
    async def start_training(
        request: TrainingRequest,
        token: HTTPAuthorizationCredentials = Depends(auth_scheme),
    ):
        # TODO: factorize using dependency injection
        user = supabase_client.auth.get_user(jwt=token.credentials)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        logger.debug(f"User: {user.user.email} ({user.user.id}) spawning server")

        active_trainings = (
            supabase_client.table("trainings")
            .select("*")
            .eq("user_id", user.user.id)
            .eq("status", "running")
            .is_("terminated_at", "null")
            .limit(10)
            .execute()
        )

        def user_is_allowlisted(user_email: str) -> bool:
            """
            Check if the user is allowlisted based on their email.
            This is a simple check against a hardcoded list of emails.
            """
            if user_email is None:
                return False
            return user_email.endswith("@phospho.ai")

        logger.info(f"Active trainings: {active_trainings.data}")

        # Check if the user is a pro user
        user_data = (
            supabase_client.table("users").select("*").eq("id", user.user.id).execute()
        )
        user_id = user.user.id
        user_plan: Literal["pro"] | None = None
        user_slug = f"{user.user.email} {user_id}"
        if user_data.data:
            user_plan = user_data.data[0].get("plan", None)

        # Validate PRO requirements for private training features
        if (request.private_mode or request.user_hf_token) and user_plan != "pro":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Private training features require PRO subscription",
            )

        # Handle timeout based on user plan. Default to 1 hours for normal users.
        timeout_seconds = BASE_TIMEOUT
        if user_plan == "pro":
            # Pro users get longer timeout
            logger.info(f"{user_slug} is a PRO user, extending timeout to 3 hours")
            timeout_seconds = PRO_TIMEOUT
        if user_is_allowlisted(user.user.email):
            logger.info(f"{user_slug} is allowlisted, extending timeout to 12 hours")
            timeout_seconds = ALLOWLISTED_TIMEOUT

        if user_plan is None and len(active_trainings.data) >= BASE_TRAININGS_LIMIT:
            logger.warning(f"{user_slug} already has an active training")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"You have already {BASE_TRAININGS_LIMIT} active trainings, please wait for one to finish or upgrade to a PRO plan",
            )
        elif user_plan == "pro" and len(active_trainings.data) >= PRO_TRAININGS_LIMIT:
            logger.warning(f"{user_slug} already has an active training")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"You have already {PRO_TRAININGS_LIMIT} active trainings, please wait for one to finish.",
            )

        # Check the user quota: max 3 trainings per month for free users, 100 per month for pro users.
        # It resets on the 1st of each month.
        current_month = datetime.now(timezone.utc).month
        user_quota = (
            supabase_client.table("trainings")
            .select("*")
            .eq("user_id", user.user.id)
            .gte(
                "requested_at",
                datetime(datetime.now(timezone.utc).year, current_month, 1),
            )
            .in_("status", ["succeeded", "running", "canceled"])
            .execute()
        )
        logger.info(
            f"{user_slug} ran {len(user_quota.data)} succeeded, running, or canceled trainings this month"
        )
        if user_is_allowlisted(user.user.email):
            logger.info(f"{user_slug} is allowlisted, skipping monthly quota check")
        elif user_plan is None and len(user_quota.data) >= BASE_MONTHLY_LIMIT:
            logger.warning(
                f"{user_slug} has reached the monthly quota of {BASE_MONTHLY_LIMIT} trainings"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"You have reached your monthly quota of {BASE_MONTHLY_LIMIT} trainings. Upgrade to a PRO plan for more trainings.",
            )
        elif user_plan == "pro" and len(user_quota.data) >= PRO_MONTHLY_LIMIT:
            logger.warning(
                f"{user_slug} has reached the monthly quota of {PRO_MONTHLY_LIMIT} trainings"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"You have reached your monthly quota of {PRO_MONTHLY_LIMIT} trainings. Limit is reset on the 1st of each month. Contact support if you need more.",
            )

        supabase_data = {
            "status": "running",
            "user_id": user.user.id,
            "used_wandb": request.wandb_api_key is not None,
        }

        request_data = request.model_dump(
            exclude_unset=True,
            exclude={"wandb_api_key", "training_params", "user_hf_token"},
        )
        if request.training_params:
            supabase_data["training_params"] = request.training_params.model_dump()
        for key, value in request_data.items():
            supabase_data[key] = value

        # We generate a run id for WandB
        wandb_run_id = generate_wandb_run_id(model_name=request.model_name)
        supabase_data["wandb_run_id"] = wandb_run_id

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
            "pi0.5": train_pi05,
            "smolvla": train_smolvla,
        }

        logger.info(f"Starting training for {request.model_type}")
        logger.info(f"Training params: {request.training_params}")

        # We pass all the parameters to the training function
        spawn_response = models_dict[request.model_type].spawn(
            **request.model_dump(exclude={"training_params"}),
            training_id=training_id,
            training_params=request.training_params,
            timeout_seconds=timeout_seconds,
            wandb_run_id=wandb_run_id,
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
        user = supabase_client.auth.get_user(jwt=token.credentials)
        if user is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        logger.debug(f"User: {user.user.email} ({user.user.id}) cancelling training")

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
                status_code=404,
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

    @web_app.post("/stripe/webhooks")
    async def stripe_webhook(request: Request):
        """
        Stripe webhook endpoint
        TODO: return a 200 status directly for Stripe and update the database asynchronously
        """
        # Get the request body
        payload = await request.body()
        sig_header = request.headers.get("stripe-signature")
        event = None

        # Get the webhook secret from environment
        endpoint_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

        try:
            event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
        except ValueError as e:
            # Invalid payload
            raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.SignatureVerificationError as e:
            # Invalid signature
            raise HTTPException(status_code=400, detail="Invalid signature")

        if (
            event["type"] == "checkout.session.completed"
            or event["type"] == "checkout.session.async_payment_succeeded"
        ):
            # Retrieve the Checkout Session from the API with line_items expanded and the metadata
            checkout_session = stripe.checkout.Session.retrieve(
                event["data"]["object"]["id"],
                expand=["line_items"],
            )
            logger.info(f"Checkout session: {checkout_session}")

            # Check if this is a subscription checkout session
            if not checkout_session.get("subscription"):
                logger.info(
                    "Checkout session does not have a subscription, ignoring event"
                )
                return {"status": "ok"}

            # Extract metadata fields
            # metadata is a dict or None, so we need to handle the case where it is None
            metadata = checkout_session.metadata or {}
            supabase_user_email = metadata.get("supabase_user_email")
            supabase_user_id = metadata.get("supabase_user_id")

            if not supabase_user_email or not supabase_user_id:
                logger.error(
                    "Checkout session metadata is missing supabase_user_email or supabase_user_id"
                )
                return {"status": "ok"}

            if checkout_session.payment_status != "unpaid":
                # Check if the user already exists in the users table.
                # This is not the default supabase auth.users table, but the public.users table
                user_data = (
                    supabase_client.table("users")
                    .select("*")
                    .eq("id", supabase_user_id)
                    .execute()
                )
                # If the user already exists, update the plan to pro, add the stripe customer id and the subscription id
                if user_data.data:
                    # User exists, update the plan to pro
                    supabase_client.table("users").update(
                        {
                            "plan": "pro",
                            "email": supabase_user_email,
                            "stripe_customer_id": checkout_session.customer,
                            "stripe_subscription_id": checkout_session.subscription,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        }
                    ).eq("id", supabase_user_id).execute()
                    logger.info(f"Updated user {supabase_user_email} to pro")
                else:
                    # In the users table, create a new user with the plan to pro, add the stripe customer id and the subscription id
                    supabase_client.table("users").insert(
                        {
                            "id": supabase_user_id,
                            "email": supabase_user_email,
                            "plan": "pro",
                            "stripe_customer_id": checkout_session.customer,
                            "stripe_subscription_id": checkout_session.subscription,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                    ).execute()
                    logger.info(f"Created new user {supabase_user_email} with plan pro")
            else:
                logger.warning(
                    "Received a checkout session with payment status unpaid!"
                )

        elif event["type"] == "customer.subscription.deleted":
            # Handle subscription cancellation/deletion
            subscription = event["data"]["object"]
            subscription_id = subscription["id"]
            customer_id = subscription["customer"]
            logger.error(
                f"Processing subscription deletion for subscription_id: {subscription_id} customer_id: {customer_id}"
            )

            # Find the user by stripe_subscription_id
            user_data = (
                supabase_client.table("users")
                .select("*")
                .eq("stripe_subscription_id", subscription_id)
                .execute()
            )

            if user_data.data:
                user = user_data.data[0]
                user_id = user["id"]

                # Update the user's plan back to free and clear subscription data
                supabase_client.table("users").update({"plan": "free"}).eq(
                    "id", user_id
                ).execute()

                logger.info(
                    f"Updated user {user_id} plan to free due to subscription cancellation"
                )
            else:
                logger.warning(f"No user found with subscription_id: {subscription_id}")

        elif event["type"] == "customer.subscription.updated":
            # Handle subscription updates (e.g., plan changes, status changes)
            subscription = event["data"]["object"]
            subscription_id = subscription["id"]
            subscription_status = subscription["status"]
            customer_id = subscription["customer"]
            logger.error(
                f"Processing subscription update for subscription_id: {subscription_id}, new status: {subscription_status}"
            )

            # Find the user by customer ID
            user_data = (
                supabase_client.table("users")
                .select("*")
                .eq("stripe_customer_id", customer_id)
                .execute()
            )

            if user_data.data:
                user = user_data.data[0]
                user_id = user["id"]

                # Handle different subscription statuses
                if subscription_status in ["canceled", "unpaid", "past_due"]:
                    # Downgrade to free plan
                    supabase_client.table("users").update(
                        {
                            "plan": "free",
                        }
                    ).eq("id", user_id).execute()

                    logger.info(
                        f"Updated user {user_id} plan to free due to subscription status: {subscription_status}"
                    )

                elif subscription_status == "active":
                    # Ensure user has pro plan
                    supabase_client.table("users").update(
                        {
                            "plan": "pro",
                        }
                    ).eq("id", user_id).execute()

                    logger.info(
                        f"Updated user {user_id} plan to pro due to active subscription"
                    )
            else:
                logger.error(f"No user found with subscription_id: {subscription_id}")

        return {"status": "ok"}

    async def get_user_and_check_quota(
        token: HTTPAuthorizationCredentials = Depends(auth_scheme),
    ) -> SupabaseUser:
        user_auth = supabase_client.auth.get_user(jwt=token.credentials)
        if not user_auth:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        # Check if the user is a pro user
        user_data = (
            supabase_client.table("users")
            .select("*")
            .eq("id", user_auth.user.id)
            .execute()
        )
        if not user_data.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        user_plan = user_data.data[0].get("plan", None)
        if user_plan != "pro":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="This endpoint is only available for PRO users",
            )

        return user_auth.user

    @web_app.post("/ai-control/chat", response_model=ChatResponse)
    async def ai_control_chat(
        chat_request: ChatRequest,
        user: SupabaseUser = Depends(get_user_and_check_quota),
    ):
        """
        Endpoint to handle AI control chat requests.
        """

        from .chat import GeminiAgent

        gemini_agent = GeminiAgent()
        return await gemini_agent.run(chat_request=chat_request)

    # Required by modal
    return web_app
