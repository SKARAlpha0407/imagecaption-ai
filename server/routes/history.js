const express = require('express');
const ImageRecord = require('../models/ImageRecord');
const { apiLimiter } = require('../middleware/rateLimiter');

const router = express.Router();

/**
 * GET /api/history
 * Fetch recent caption generation history for a user or globally.
 * Query params: userId, limit (default 20)
 */
router.get('/', apiLimiter, async (req, res, next) => {
  try {
    const userId = req.query.userId || null;
    const limit = Math.min(parseInt(req.query.limit) || 20, 100);

    const query = userId ? { userId } : {};
    const docs = await ImageRecord.find(query, '-__v')
      .sort({ createdAt: -1 })
      .limit(limit)
      .lean();

    res.json({
      success: true,
      data: docs,
      count: docs.length,
      filter: userId ? { userId } : 'global',
    });
  } catch (err) {
    next(err);
  }
});

/**
 * DELETE /api/history/:id
 * Remove a history record.
 */
router.delete('/:id', apiLimiter, async (req, res, next) => {
  try {
    const doc = await ImageRecord.findByIdAndDelete(req.params.id);
    if (!doc) {
      return res.status(404).json({ success: false, error: 'Record not found' });
    }
    res.json({ success: true, message: 'Record deleted', deletedId: doc._id });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
