import crypto from "node:crypto";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";
import express from "express";
import multer from "multer";
import {
  IdentificationServiceError,
  identifyWithLocalModel,
  identifyWithPlantNet,
} from "./layer1.js";

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const allowedOrgans = new Set(["auto", "leaf", "flower", "fruit", "bark"]);

const upload = multer({
  storage: multer.diskStorage({
    destination: os.tmpdir(),
    filename: (_request, file, callback) => {
      const extension = file.mimetype === "image/png" ? ".png" : ".jpg";
      callback(null, `flora-${crypto.randomUUID()}${extension}`);
    },
  }),
  limits: { fileSize: 10 * 1024 * 1024, files: 1 },
  fileFilter: (_request, file, callback) => {
    if (!["image/jpeg", "image/png"].includes(file.mimetype)) {
      callback(new multer.MulterError("LIMIT_UNEXPECTED_FILE", file.fieldname));
      return;
    }
    callback(null, true);
  },
});

function receiveImage(request, response, next) {
  upload.single("image")(request, response, (error) => {
    if (error) {
      const message =
        error.code === "LIMIT_FILE_SIZE"
          ? "Image must be 10MB or smaller."
          : "Upload a JPEG or PNG image.";
      response.status(400).json({ status: "error", message });
      return;
    }
    if (!request.file) {
      response.status(400).json({ status: "error", message: "An image is required." });
      return;
    }
    next();
  });
}

function selectedOrgan(request) {
  const organ = String(request.body.organ ?? "auto").toLowerCase();
  return allowedOrgans.has(organ) ? organ : "auto";
}

async function removeTemporaryFile(file) {
  if (!file?.path) return;
  try {
    await fs.unlink(file.path);
  } catch (error) {
    if (error.code !== "ENOENT") {
      console.warn("Could not remove temporary upload:", error.message);
    }
  }
}

function identificationRoute(handler) {
  return async (request, response, next) => {
    try {
      response.json(await handler(request.file, selectedOrgan(request)));
    } catch (error) {
      next(error);
    } finally {
      await removeTemporaryFile(request.file);
    }
  };
}

app.disable("x-powered-by");
app.use(express.static(__dirname));

app.get("/health", (_request, response) => {
  response.json({
    status: "ok",
    plantnetConfigured: Boolean(process.env.PLANTNET_API_KEY),
    trefleConfigured: Boolean(process.env.TREFLE_TOKEN),
  });
});

app.post(
  "/identify/plantnet",
  receiveImage,
  identificationRoute(identifyWithPlantNet),
);
app.post(
  "/identify/local",
  receiveImage,
  identificationRoute(identifyWithLocalModel),
);

app.use((error, _request, response, _next) => {
  if (error instanceof IdentificationServiceError) {
    response.status(503).json({ status: "error", message: error.message });
    return;
  }
  console.error("Unexpected server error:", error);
  response.status(500).json({ status: "error", message: "Something went wrong." });
});

const port = Number(process.env.PORT) || 3000;
const server = app.listen(port, () => {
  console.log(`flora. is listening at http://localhost:${port}`);
});

export { app, server };
