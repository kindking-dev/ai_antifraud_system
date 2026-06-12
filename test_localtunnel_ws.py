import asyncio
import websockets

async def test_ws():
    try:
        async with websockets.connect(
            "wss://sentinel-mobile-app.loca.lt/api/v1/ws/telemetry/inspector",
            extra_headers={"Bypass-Tunnel-Reminder": "true"}
        ) as ws:
            print("Connected via WSS!")
            await asyncio.sleep(2)
            print("Done")
    except Exception as e:
        print("Error:", e)

asyncio.run(test_ws())
