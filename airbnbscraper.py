from apify_client import ApifyClient

# Initialize the ApifyClient with your API token
client = ApifyClient("<YOUR_API_TOKEN>")

# Prepare the Actor input
run_input = {
    "locationQueries": ["London"],
    "startUrls": [],
    "enrichUserProfiles": None,
    "checkIn": None,
    "checkOut": None,
    "locale": "en-US",
    "priceMin": None,
    "currency": "USD",
    "priceMax": None,
    "minBeds": None,
    "minBedrooms": None,
    "minBathrooms": None,
    "adults": None,
    "children": None,
    "infants": None,
    "pets": None,
}

# Run the Actor and wait for it to finish
run = client.actor("GsNzxEKzE2vQ5d9HN").call(run_input=run_input)

# Fetch and print Actor results from the run's dataset (if there are any)
for item in client.dataset(run["defaultDatasetId"]).iterate_items():
    print(item)