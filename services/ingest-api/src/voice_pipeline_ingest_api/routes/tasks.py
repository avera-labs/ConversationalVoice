from fastapi import APIRouter, HTTPException, status
from voice_pipeline_task_client import (
    TaskPublicationError,
    UnknownTaskNameError,
)

from ..dependencies import TaskPublisherDependency
from ..schemas import TriggerTaskRequest, TriggerTaskResponse

router = APIRouter(prefix="/v1/tasks", tags=["tasks"])


@router.post(
    "/trigger",
    response_model=TriggerTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def trigger_task(
    request: TriggerTaskRequest,
    publisher: TaskPublisherDependency,
) -> TriggerTaskResponse:
    """Publish a registered Celery task with one UUID argument."""
    try:
        task_id = publisher.publish_registered(request.task_name, request.id)
    except UnknownTaskNameError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Task name is not registered.",
        ) from exc
    except TaskPublicationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task publication is unavailable.",
        ) from exc

    return TriggerTaskResponse(
        task_name=request.task_name,
        id=request.id,
        task_id=task_id,
    )
