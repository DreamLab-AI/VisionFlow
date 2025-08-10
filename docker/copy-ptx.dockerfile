# PTX File Copy Instructions for Dockerfile
# Add this to your Dockerfile to copy pre-compiled PTX files

# Copy pre-compiled PTX files to avoid compilation during build
COPY src/utils/ptx/*.ptx /app/src/utils/ptx/

# Ensure PTX directory exists and has correct permissions
RUN mkdir -p /app/src/utils/ptx && chmod -R 755 /app/src/utils/ptx