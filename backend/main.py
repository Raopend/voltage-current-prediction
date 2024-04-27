import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import state

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(state.router)


if __name__ == "__main__":
    uvicorn.run(
        app="main:app", host="0.0.0.0", port=8000, log_level="debug", reload=True
    )
