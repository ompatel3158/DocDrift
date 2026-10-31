import express from "express";

const app = express();
const port = process.env.PORT || 3000;
const databaseUrl = process.env.DATABASE_URL;

app.get("/health", (req, res) => {
  res.json({ status: "ok", port, databaseUrl });
});

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
