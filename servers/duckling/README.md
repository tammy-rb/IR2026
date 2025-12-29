# Duckling Server (Docker)

This directory contains a Docker Compose setup for running the **Duckling** time-extraction server locally.
Duckling is used to extract structured temporal expressions from natural language queries.

---

## Requirements

- Docker
- Docker Compose (included with Docker Desktop)

---

## Starting the Duckling Server

Run the following command **from this directory**:

```bash
docker compose up
```

This starts the Duckling server and exposes it at:

- http://localhost:8000

To run the server in the background:

```bash
docker compose up -d
```

---

## Stopping the Duckling Server

To stop and remove the running container:

```bash
docker compose down
```

---

## Connecting to the Duckling Server

Duckling exposes a single HTTP endpoint for parsing text:

- `POST http://localhost:8000/parse`

### Example Request

```bash
curl -X POST http://localhost:8000/parse \
  --data "locale=en_US&text=tomorrow at 8pm"
```

### Example Response

```json
[
  {
    "body": "tomorrow at 8pm",
    "start": 0,
    "end": 15,
    "dim": "time",
    "latent": false,
    "value": {
      "type": "value",
      "value": "2020-09-28T20:00:00.000-07:00",
      "grain": "hour"
    }
  }
]
```

Duckling returns structured temporal information, including:
- matched text span (`body`)
- normalized timestamps (`value.value` or `value.from`/`value.to`)
- temporal resolution (`value.grain`)
- entity type (`value.type`: `value` vs `interval`)

---

## Usage in This Project

In this project, Duckling is used as a standalone temporal extraction service:
- queried from Python over HTTP
- integrated into the temporal analysis stage of the RAG pipeline
- only the `time` dimension is used

---

## Notes

- The Duckling server runs only when explicitly started.
- No automatic restart policy is configured.
- Port **8000** must be available on the host machine.

---

## Additional Resources

- *Using Duckling to Extract Dates and Times in Your Rasa Chatbot*  
  https://medium.com/@adboio/using-duckling-to-extract-dates-and-times-in-your-rasa-chatbot-7687f4fde2e0
