# Sample Service

A backend service for demonstration.

## Quickstart

Run the following command to start the application:

```bash
npm run dev
```

Run test suite:
```bash
npm test
```

## Configuration

Settings are stored in `config/database.json`. Make sure you create this file.

Set your environment variables in `.env`:
- `MONGO_URL`: Connection string for MongoDB
- `PORT_NUM`: Server port number

## Build & Lint

To check linting before committing:
```bash
make lint
```
