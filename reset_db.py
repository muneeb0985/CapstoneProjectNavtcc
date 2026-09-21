import asyncio
import asyncpg

async def reset():
    # Update password if yours is different from 'admin123'
    conn = await asyncpg.connect("postgresql://postgres:123@localhost:5432/Contract")
    await conn.execute("DROP TABLE IF EXISTS contracts CASCADE;")
    print("SUCCESS: Old contracts table dropped successfully!")
    await conn.close()

asyncio.run(reset())