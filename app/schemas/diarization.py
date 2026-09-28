"""Speaker diarization schemas."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SpeakerTurn(BaseModel):
    """A continuous stretch of speech by one speaker."""

    speaker_label: str = Field(description='Stable meeting-wide label, e.g. "Person 1".')
    raw_label: str = Field(description='Label emitted by the model, e.g. "SPEAKER_00".')
    start: float = Field(description="Seconds from the start of the processed audio.")
    end: float
    duration: float
    is_overlap: bool = Field(description="True if another speaker talks during this turn.")


class OverlapRegion(BaseModel):
    """A span where two or more speakers talk at once."""

    start: float
    end: float
    speakers: list[str]


class DiarizationResult(BaseModel):
    turns: list[SpeakerTurn] = Field(description="Sorted by start time.")
    num_speakers: int
    overlap_regions: list[OverlapRegion]
    model_name: str
    processing_time_ms: int


class SpeakersResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "meeting_id": "3f8b6f0e-2c1d-4d6a-9d3e-6c2b1a0f9e7d",
                    "num_speakers": 2,
                    "speakers": ["Person 1", "Person 2"],
                    "turns": [
                        {
                            "speaker_label": "Person 1",
                            "raw_label": "SPEAKER_01",
                            "start": 0.52,
                            "end": 7.84,
                            "duration": 7.32,
                            "is_overlap": False,
                        },
                        {
                            "speaker_label": "Person 2",
                            "raw_label": "SPEAKER_00",
                            "start": 7.61,
                            "end": 15.2,
                            "duration": 7.59,
                            "is_overlap": True,
                        },
                    ],
                    "overlap_regions": [
                        {"start": 7.61, "end": 7.84, "speakers": ["Person 1", "Person 2"]}
                    ],
                    "model_name": "pyannote/speaker-diarization-3.1",
                }
            ]
        }
    )

    meeting_id: UUID
    num_speakers: int
    speakers: list[str]
    turns: list[SpeakerTurn]
    overlap_regions: list[OverlapRegion]
    model_name: str
