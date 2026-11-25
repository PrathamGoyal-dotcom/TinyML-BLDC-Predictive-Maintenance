"""Research baseline evaluation endpoints (/api/v1/baseline/*)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.db.repository import RecordingsRepository
from app.ml.baseline import (
    evaluate_dudu_baseline,
    generate_synthetic_bldc_dataset,
    load_csv_dataset,
    load_db_dataset,
)
from app.schemas import BaselineEvaluateRequest, BaselineEvaluateResponse

router = APIRouter(prefix="/api/v1/baseline", tags=["Research Baselines"])


def get_repo(request: Request) -> RecordingsRepository:
    return request.app.state.repo


@router.post(
    "/evaluate",
    response_model=BaselineEvaluateResponse,
    summary="Run leakage-free recording_id-grouped K-fold CV on the 28-feature DUDU-BLDC baseline",
)
def evaluate_baseline(
    payload: BaselineEvaluateRequest,
    repo: RecordingsRepository = Depends(get_repo),
) -> BaselineEvaluateResponse:
    """Evaluate the 28-feature tabular baseline on synthetic data, exported CSV, or SQLite DB."""
    try:
        if payload.source == "csv":
            if not payload.csv_path:
                raise ValueError("csv_path is required when source='csv'")
            X, y, groups = load_csv_dataset(payload.csv_path)
        elif payload.source == "database":
            X, y, groups = load_db_dataset(repo)
        else:
            X, y, groups, _ = generate_synthetic_bldc_dataset(
                n_recordings=payload.n_recordings,
                windows_per_recording=payload.windows_per_recording,
                snr_db=payload.snr_db,
                seed=payload.seed,
            )

        return evaluate_dudu_baseline(
            X,
            y,
            groups,
            source=payload.source,
            n_splits=payload.n_splits,
            seed=payload.seed,
            model_type=payload.model_type,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
