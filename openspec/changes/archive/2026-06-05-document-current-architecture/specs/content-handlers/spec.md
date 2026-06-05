## ADDED Requirements

### Requirement: Chain of Responsibility pattern

The system SHALL implement a Chain of Responsibility pattern for content extraction. Each handler SHALL inherit from BaseContentTypeHandler and implement can_handle() and stream_content(). The chain SHALL be traversed in order: PDF -> MP4 -> MP3 -> Text. The first handler whose can_handle() returns True SHALL be used for extraction.

#### Scenario: Handler chain traversal
- **WHEN** a .pdf file is processed
- **THEN** PDFContentTypeHandler.can_handle() SHALL return True and the handler SHALL be used for extraction, without falling through to subsequent handlers

#### Scenario: Fallthrough to next handler
- **WHEN** a file type is not handled by the current handler
- **THEN** the request SHALL be passed to the next handler in the chain via the next_handler parameter

#### Scenario: No handler matches
- **WHEN** no handler in the chain can process the file
- **THEN** None SHALL be returned

### Requirement: MIME type determination

Each handler SHALL provide a get_mime_type() method. If the handler class defines a MIME_TYPE class variable, that value SHALL be returned. Otherwise, the system SHALL use mimetypes.guess_type() to determine the MIME type from the file extension. The determined MIME type SHALL flow through Redis job metadata to downstream workers (e.g., whisperx worker for audio/video files).

#### Scenario: Fixed MIME type from class variable
- **WHEN** PDFContentTypeHandler.get_mime_type() is called
- **THEN** it SHALL return "application/pdf"

#### Scenario: Guessed MIME type from extension
- **WHEN** TextContentTypeHandler.get_mime_type() is called with a .md or .txt file
- **THEN** it SHALL return the MIME type guessed from the file extension

### Requirement: PDF content extraction with OCR fallback

PDFContentTypeHandler SHALL handle files with MIME type application/pdf. It SHALL attempt direct text extraction via pdfplumber. If a page has insufficient extractable text, the system SHALL render the page to an image via pdf2image and send it to the OCR worker via send_image_to_ocr(). OCR results SHALL be validated; bad OCR results SHALL trigger re-processing. The handler SHALL support synchronous OCR within the page loop.

#### Scenario: Digital PDF with extractable text
- **WHEN** a PDF page has extractable text via pdfplumber
- **THEN** the text is extracted directly without OCR

#### Scenario: Scanned PDF with OCR fallback
- **WHEN** a PDF page has insufficient extractable text (below threshold)
- **THEN** the page is rendered to an image, preprocessed, and sent to the OCR worker

#### Scenario: OCR validation
- **WHEN** OCR returns text that fails validation (is_bad_ocr returns True)
- **THEN** the handler SHALL request re-processing of that page

### Requirement: Media file transcription

The system SHALL support audio and video file transcription via WhisperX. MP4ContentTypeHandler SHALL handle video files (.mp4, .mov, .mkv) with MIME type video/mp4. MP3ContentTypeHandler SHALL handle audio files (.mp3, .wav, .m4a, .aac, .flac) with MIME type guessed from extension. Both handlers SHALL delegate transcription to the whisperx worker via send_media_to_whisperx().

#### Scenario: MP4 file transcription
- **WHEN** an .mp4 file is processed
- **THEN** MP4ContentTypeHandler SHALL delegate to the whisperx worker and return the transcribed text

#### Scenario: MP3 file transcription
- **WHEN** an .mp3 file is processed
- **THEN** MP3ContentTypeHandler SHALL delegate to the whisperx worker and return the transcribed text

#### Scenario: Whisper MIME type flow
- **WHEN** a media handler sends a job to the whisperx worker
- **THEN** the MIME type SHALL be included in the Redis job metadata so the whisper server uses the correct Content-Type

### Requirement: Plain text file reading

TextContentTypeHandler SHALL handle .txt, .md, .html, and .htm files. It SHALL detect character encoding and read the file content as plain text.

#### Scenario: Text file reading
- **WHEN** a .txt file is processed
- **THEN** TextContentTypeHandler SHALL read the file content and yield it as a stream
