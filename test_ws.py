import asyncio
import websockets

async def test_ws():
    try:
        async with websockets.connect("ws://127.0.0.1:8000/api/v1/ws/telemetry/inspector", origin="https://sentinel-mobile-app.loca.lt") as ws:
            print("Connected!")
            await ws.recv()
    except Exception as e:
        print("Error:", e)

asyncio.run(test_ws())
