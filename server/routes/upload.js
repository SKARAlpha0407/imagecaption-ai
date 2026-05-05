const express = require('express');
const axios = require('axios');
const FormData = require('form-data');
const ImageRecord = require('../models/ImageRecord');
const { upload } = require('../middleware/upload');
const { uploadLimiter } = require('../middleware/rateLimiter');

const router = express.Router();

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

/**
 * POST /api/upload
 * Accepts multipart image upload, proxies to FastAPI ML service,
 * stores result in MongoDB, returns JSON to client.
 */
router.post('/', uploadLimiter, upload.single('image'), async (req, res, next) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'No image file provided' });
    }

    // Build form-data for FastAPI
    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    // Proxy to ML service
    const mlResponse = await axios.post(`${ML_SERVICE_URL}/predict`, form, {
      headers: form.getHeaders(),
      timeout: 30000,
      maxBodyLength: 10 * 1024 * 1024,
      maxContentLength: 10 * 1024 * 1024,
    });

    const mlData = mlResponse.data;
    if (!mlData.success) {
      return res.status(502).json({ success: false, error: 'ML service prediction failed' });
    }

    // Save to MongoDB
    const record = new ImageRecord({
      imageUrl: req.file.originalname,
      predictedCaption: mlData.caption,
      actualCaptions: [],
      userId: req.body.userId || null,
      confidence: mlData.confidence || 0,
      modelVersion: mlData.model || 'vgg16_lstm',
    });
    await record.save();

    return res.status(200).json({
      success: true,
      data: {
        id: record._id,
        caption: mlData.caption,
        confidence: mlData.confidence,
        imageName: req.file.originalname,
        createdAt: record.createdAt,
      },
    });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
