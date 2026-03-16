package com.selfhosted.rag.common.utils;

import com.selfhosted.rag.common.config.AppConfig;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;

import java.io.File;
import java.io.IOException;
import java.text.Normalizer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Text processing utility functions.
 * Maps to Python: utils/text_utils.py
 */
@ApplicationScoped
public class TextUtils {

    @Inject
    AppConfig config;

    private static final Pattern LATIN_PATTERN = Pattern.compile("\\p{IsLatin}");
    private static final Pattern VISIBLY_CORRUPT_PATTERN = Pattern.compile("[âã¢£™žÂÃ]");
    private static final Pattern VISIBLY_CORRUPT_LATIN_EXT_PATTERN = Pattern.compile("[âã¢£™žœÂÃ]");

    /**
     * Return fraction of characters that are Latin letters.
     */
    public double latinScriptRatio(String text) {
        if (text == null || text.isEmpty()) {
            return 0.0;
        }
        Matcher matcher = LATIN_PATTERN.matcher(text);
        int count = 0;
        while (matcher.find()) {
            count++;
        }
        return (double) count / text.length();
    }

    /**
     * Check if text contains corruption indicators.
     */
    public boolean isVisiblyCorrupt(String text) {
        if (text == null) return false;
        Pattern pattern = config.isAllowLatinExtended() ? VISIBLY_CORRUPT_PATTERN : VISIBLY_CORRUPT_LATIN_EXT_PATTERN;
        return pattern.matcher(text).find();
    }

    /**
     * Check if text appears to be gibberish.
     */
    public boolean isGibberish(String text) {
        if (text == null || text.isBlank()) {
            return true;
        }

        String normalized = Normalizer.normalize(text, config.isAllowLatinExtended() ? Normalizer.Form.NFC : Normalizer.Form.NFKD);
        StringBuilder printableBuilder = new StringBuilder();
        for (int i = 0; i < normalized.length(); i++) {
            char c = normalized.charAt(i);
            if (!Character.isISOControl(c)) {
                printableBuilder.append(c);
            }
        }
        String printable = printableBuilder.toString();
        int total = printable.length();
        if (total == 0) return true;

        if (config.isAllowLatinExtended() && latinScriptRatio(printable) >= config.getLatinScriptMinRatio()) {
            int nonAlpha = 0;
            for (int i = 0; i < printable.length(); i++) {
                char c = printable.charAt(i);
                if (Character.getType(c) == Character.NON_SPACING_MARK) continue;
                if (!(Character.isLetter(c) || isCommonPunctuation(c))) {
                    nonAlpha++;
                }
            }
            double ratio = (double) nonAlpha / Math.max(1, total);
            return ratio > 0.75;
        }

        int nonAlpha = 0;
        for (int i = 0; i < printable.length(); i++) {
            char c = printable.charAt(i);
            if (config.isAllowLatinExtended() && Character.getType(c) == Character.NON_SPACING_MARK) continue;
            if (!(Character.isLetter(c) || Character.isWhitespace(c))) {
                nonAlpha++;
            }
        }
        double ratio = (double) nonAlpha / total;
        return ratio > 0.6;
    }

    private boolean isCommonPunctuation(char c) {
        return Character.isWhitespace(c) || "-\t\n\r–—·.,;:()[]'\"".indexOf(c) >= 0;
    }

    /**
     * Check if text is invalid for processing.
     */
    public boolean isInvalidText(String text) {
        if (text == null || text.strip().length() < 20) {
            return true;
        }

        if (config.isAllowLatinExtended()) {
            int printableCount = 0;
            for (int i = 0; i < text.length(); i++) {
                if (!Character.isISOControl(text.charAt(i))) {
                    printableCount++;
                }
            }
            double ratio = (double) printableCount / text.length();
            if (ratio < 0.6) return true;
            return latinScriptRatio(text) < config.getLatinScriptMinRatio();
        }

        return !isMostlyPrintableAscii(text, 0.75);
    }

    /**
     * Check if text is low quality based on token count.
     * Replicates Python: is_low_quality(text, tokenizer, min_tokens=5)
     */
    public boolean isLowQuality(String text, int tokenCount) {
        return tokenCount < 5;
    }

    private boolean isMostlyPrintableAscii(String text, double threshold) {
        if (text == null || text.isEmpty()) return false;
        int printableCount = 0;
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c >= 32 && c <= 126 || Character.isWhitespace(c)) {
                printableCount++;
            }
        }
        return (double) printableCount / text.length() >= threshold;
    }

    /**
     * Validates whether a file is a structurally valid PDF.
     */
    public boolean isValidPdf(String path) {
        File file = new File(path);
        if (!file.exists() || !file.isFile()) return false;

        try (PDDocument document = Loader.loadPDF(file)) {
            return document.getNumberOfPages() > 0;
        } catch (IOException e) {
            return false;
        }
    }
}
