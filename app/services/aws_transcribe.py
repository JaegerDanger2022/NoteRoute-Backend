# Phase 4: AWS Transcribe wrapper
# Implemented during LangGraph pipeline phase.


async def start_transcription_job(s3_key: str, job_name: str) -> str:
    """Start an AWS Transcribe job. Returns the job name."""
    raise NotImplementedError("AWS Transcribe service not yet implemented (Phase 4)")


async def get_transcript(job_name: str) -> dict:
    """Poll AWS Transcribe job until complete. Returns transcript text and confidence."""
    raise NotImplementedError("AWS Transcribe service not yet implemented (Phase 4)")
