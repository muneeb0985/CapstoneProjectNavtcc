import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
pool = None

async def init_db():
    global pool
    pool = await asyncpg.create_pool(
        dsn=DATABASE_URL,
        min_size=2,
        max_size=10
    )

async def close_db():
    global pool
    if pool:
        await pool.close()