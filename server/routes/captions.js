const express = require('express');
const ImageRecord = require('../models/ImageRecord');
const { apiLimiter } = require('../middleware/rateLimiter');

const router = express.Router();

/**
 * GET /api/captions
 * List all generated captions with optional pagination.
 */
router.get('/captions', apiLimiter, async (req, res, next) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 20;
    const skip = (page - 1) * limit;

    const [docs, total] = await Promise.all([
      ImageRecord.find({}, '-__v')
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(limit)
        .lean(),
      ImageRecord.countDocuments(),
    ]);

    res.json({
      success: true,
      data: docs,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    });
  } catch (err) {
    next(err);
  }
});

/**
 * GET /api/captions/:id
 * Get a single caption record by MongoDB ID.
 */
router.get('/captions/:id', apiLimiter, async (req, res, next) => {
  try {
    const doc = await ImageRecord.findById(req.params.id, '-__v').lean();
    if (!doc) {
      return res.status(404).json({ success: false, error: 'Caption not found' });
    }
    res.json({ success: true, data: doc });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
