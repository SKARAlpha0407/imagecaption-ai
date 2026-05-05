const mongoose = require('mongoose');

/**
 * ImageRecord Schema
 * Stores uploaded image metadata, predicted caption, and optional actual captions.
 */
const ImageRecordSchema = new mongoose.Schema({
  imageUrl: {
    type: String,
    required: true,
    trim: true,
  },
  predictedCaption: {
    type: String,
    required: true,
    trim: true,
  },
  actualCaptions: {
    type: [String],
    default: [],
  },
  userId: {
    type: String,
    default: null,
    index: true,
  },
  confidence: {
    type: Number,
    default: 0,
  },
  modelVersion: {
    type: String,
    default: 'vgg16_lstm',
  },
}, {
  timestamps: true, // adds createdAt and updatedAt
});

// Index for querying recent history efficiently
ImageRecordSchema.index({ createdAt: -1 });
ImageRecordSchema.index({ userId: 1, createdAt: -1 });

module.exports = mongoose.model('ImageRecord', ImageRecordSchema);
