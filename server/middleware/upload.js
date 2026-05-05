const multer = require('multer');
const path = require('path');

/**
 * Multer configuration for image uploads.
 * Stores in memory (for proxying to FastAPI); disk storage optional below.
 */
const storage = multer.memoryStorage();

const fileFilter = (req, file, cb) => {
  const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
  if (allowedTypes.includes(file.mimetype)) {
    cb(null, true);
  } else {
    cb(new Error(`Invalid file type: ${file.mimetype}. Allowed: JPG, PNG, WebP`), false);
  }
};

const limits = {
  fileSize: parseInt(process.env.MAX_FILE_SIZE || '5242880', 10), // 5MB
};

const upload = multer({ storage, fileFilter, limits });

module.exports = { upload };
