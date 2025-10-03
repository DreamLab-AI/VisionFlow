import { useState, useEffect, useRef, useCallback } from 'react';
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import * as THREE from 'three';
import { createLogger } from '@/utils/loggerConfig';

const logger = createLogger('useHeadTracking');

let faceLandmarker: FaceLandmarker | undefined;
let lastVideoTime = -1;

const SMOOTHING_FACTOR = 0.15; // Smoothing for less jittery movement

export function useHeadTracking() {
  const [isEnabled, setIsEnabled] = useState(false);
  const [isTracking, setIsTracking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [headPosition, setHeadPosition] = useState<THREE.Vector2 | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const animationFrameId = useRef<number | null>(null);
  const smoothedPosition = useRef(new THREE.Vector2(0, 0));

  const initialize = useCallback(async () => {
    if (faceLandmarker) return;
    try {
      logger.info('Initializing MediaPipe Face Landmarker...');
      const filesetResolver = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm'
      );
      faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: `/models/face_landmarker.task`,
          delegate: 'GPU',
        },
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
        runningMode: 'VIDEO',
        numFaces: 1,
      });
      logger.info('Face Landmarker initialized successfully');
    } catch (e: any) {
      logger.error('Failed to initialize Face Landmarker', e);
      setError('Failed to load head tracking model. Please check your network connection.');
    }
  }, []);

  const predictWebcam = useCallback(() => {
    if (!videoRef.current || !faceLandmarker || !videoRef.current.srcObject) {
        if (animationFrameId.current) cancelAnimationFrame(animationFrameId.current);
        return;
    }

    const video = videoRef.current;
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const results = faceLandmarker.detectForVideo(video, Date.now());

      if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        // Use nose tip (landmark 1) as a stable point for head position
        const noseTip = results.faceLandmarks[0][1];
        if (noseTip) {
          // Normalize position to [-1, 1] range
          // MediaPipe gives normalized coordinates [0, 1]
          // We map x:[0,1] -> [-1,1] and y:[0,1] -> [1,-1] (inverted y)
          const newPos = new THREE.Vector2(
            (noseTip.x - 0.5) * 2,
            -(noseTip.y - 0.5) * 2
          );

          // Apply smoothing (Lerp)
          smoothedPosition.current.lerp(newPos, SMOOTHING_FACTOR);
          setHeadPosition(smoothedPosition.current.clone());
        }
      } else {
        setHeadPosition(null);
      }
    }

    animationFrameId.current = requestAnimationFrame(predictWebcam);
  }, []);

  const start = useCallback(async () => {
    if (isTracking) return;
    setError(null);

    await initialize();
    if (!faceLandmarker) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false,
      });

      if (!videoRef.current) {
        const video = document.createElement('video');
        video.style.display = 'none';
        video.autoplay = true;
        video.muted = true;
        video.playsInline = true;
        document.body.appendChild(video);
        videoRef.current = video;
      }

      videoRef.current.srcObject = stream;
      videoRef.current.addEventListener('loadeddata', () => {
        videoRef.current?.play();
        setIsTracking(true);
        predictWebcam();
      });
    } catch (err: any) {
      logger.error('Failed to get webcam access', err);
      setError('Webcam access denied. Please allow camera permissions to use head tracking.');
      setIsEnabled(false);
    }
  }, [isTracking, initialize, predictWebcam]);

  const stop = useCallback(() => {
    if (!isTracking && !videoRef.current?.srcObject) return;

    if (animationFrameId.current) {
      cancelAnimationFrame(animationFrameId.current);
      animationFrameId.current = null;
    }

    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }

    setIsTracking(false);
    setHeadPosition(null);
    smoothedPosition.current.set(0, 0);
  }, [isTracking]);

  useEffect(() => {
    if (isEnabled) {
      start();
    } else {
      stop();
    }
  }, [isEnabled, start, stop]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stop();
      if (videoRef.current) {
        document.body.removeChild(videoRef.current);
        videoRef.current = null;
      }
      // Do not close faceLandmarker, it can be reused across the app lifecycle
    };
  }, [stop]);

  return { isEnabled, setIsEnabled, isTracking, headPosition, error };
}
