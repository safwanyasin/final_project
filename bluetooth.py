import time
import bleak

# Set your phone's Bluetooth address (you can find it in your phone's Bluetooth settings)
phone_address = "2c:be:eb:19:62:21"

# Set the threshold for weak signal strength
threshold = -80  # Adjust as needed

async def discover():
    devices = await bleak.discover()
    return devices

async def connect(device):
    async with bleak.BleakClient(device.address) as client:
        await client.connect()
        return client

async def get_signal_strength(client):
    rssi = await client.get_rssi()
    return rssi

async def monitor_signal():
    while True:
        devices = await discover()

        for device in devices:
            if device.address == phone_address:
                client = await connect(device)
                signal_strength = await get_signal_strength(client)

                print(f"Signal strength for {device.name}: {signal_strength} dBm")

                if signal_strength < threshold:
                    print("Weak signal! Take action.")

        time.sleep(10)  # Adjust the interval as needed

if __name__ == "__main__":
    loop = bleak._loop._get_running_loop()
    loop.run_until_complete(monitor_signal())
