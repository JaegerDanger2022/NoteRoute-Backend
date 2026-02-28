from fastapi import APIRouter

from app.api.v1 import admin, auth, billing, deliver, gdocs, integrations, process, routes, slots, sources, trello, users, voice

router = APIRouter(prefix="/api/v1")

router.include_router(auth.router)
router.include_router(users.router)
router.include_router(sources.router)
router.include_router(slots.router)
router.include_router(routes.router)
router.include_router(voice.router)
router.include_router(integrations.router)
router.include_router(process.router)
router.include_router(deliver.router)
router.include_router(gdocs.router)
router.include_router(trello.router)
router.include_router(admin.router)
router.include_router(billing.router)
