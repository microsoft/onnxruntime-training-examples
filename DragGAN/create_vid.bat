
ffmpeg -framerate 30 -i ./out/training/iter_%%3d.png -c:v libx264 -pix_fmt yuv420p ./out/training/_training.mp4