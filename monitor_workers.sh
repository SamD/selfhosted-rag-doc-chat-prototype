#!/bin/bash
export PYTHONPATH=$(pwd)/doc-ingest-chat
echo "🚀 Starting OCR Worker..."
python3 -m run_ocr_worker > ocr_worker.log 2>&1 &
OCR_PID=$!
echo "🚀 Starting Gatekeeper Worker..."
python3 -m workers.gatekeeper_worker > gatekeeper_worker.log 2>&1 &
GK_PID=$!

echo "🛰️ Workers started: OCR=$OCR_PID, GK=$GK_PID"
echo "⏳ Waiting 60 seconds or until any dies..."

for i in {1..60}; do
    if ! kill -0 $OCR_PID 2>/dev/null; then
        echo "💥 OCR Worker (PID $OCR_PID) died with status $?"
        break
    fi
    if ! kill -0 $GK_PID 2>/dev/null; then
        echo "💥 Gatekeeper Worker (PID $GK_PID) died with status $?"
        break
    fi
    sleep 1
done

echo "🛑 Cleaning up..."
kill $OCR_PID 2>/dev/null
kill $GK_PID 2>/dev/null
echo "✅ Finished."
