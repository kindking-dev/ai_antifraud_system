import requests

payload = {
    "user_id": "live_mobile_user_01",
    "features": {
        "duration_ms_mean": 250.0,
        "duration_ms_std": 10.0,
        "duration_ms_max": 300.0,
        "length_px_mean": 400.0,
        "length_px_std": 15.0,
        "length_px_max": 450.0,
        "velocity_mean": 1.5,
        "velocity_std": 0.2,
        "velocity_max": 2.0,
        "median_pressure_mean": 0.5,
        "median_pressure_std": 0.05,
        "median_pressure_max": 0.6,
        "median_area_mean": 5.0,
        "median_area_std": 0.5,
        "median_area_max": 6.0
    }
}

headers = {"X-API-KEY": "DEV-MASTER-KEY", "Content-Type": "application/json"}
resp = requests.post("http://localhost:8000/api/v1/set-baseline", json=payload, headers=headers)
print(resp.json())
