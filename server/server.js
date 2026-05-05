require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');

const logger = require('./utils/logger');
const errorHandler = require('./middleware/errorHandler');

const uploadRoutes = require('./routes/upload');
const captionsRoutes = require('./routes/captions');
const historyRoutes = require('./routes/history');

const app = express();
const PORT = process.env.PORT || 5001;
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/imagecaption';

// Security & middleware
app.use(helmet());
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:5173'],
  credentials: true
}));
app.use(express.json());
app.use(morgan('combined', { stream: { write: (msg) => logger.info(msg.trim()) } }));

// Health check
app.get('/health', (req, res) => {
  const dbState = mongoose.connection.readyState;
  const dbStatus = dbState === 1 ? 'connected' : 'disconnected';
  res.json({ status: 'ok', db: dbStatus, timestamp: new Date().toISOString() });
});

// API routes
app.use('/api/upload', uploadRoutes);
app.use('/api/captions', captionsRoutes);
app.use('/api/history', historyRoutes);

// Error handling
app.use(errorHandler);

// 404
app.use((req, res) => {
  res.status(404).json({ success: false, error: 'Route not found' });
});

// Connect to MongoDB
mongoose.connect(MONGO_URI)
  .then(() => {
    logger.info(`[DB] MongoDB connected: ${MONGO_URI}`);
    app.listen(PORT, () => {
      logger.info(`[SERVER] Express running on port ${PORT}`);
    });
  })
  .catch((err) => {
    logger.error(`[DB] MongoDB connection error: ${err.message}`);
    process.exit(1);
  });
