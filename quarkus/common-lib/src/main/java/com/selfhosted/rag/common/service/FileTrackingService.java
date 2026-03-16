package com.selfhosted.rag.common.service;

import com.selfhosted.rag.common.config.AppConfig;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Service for tracking ingested and failed files.
 * Maps to Python: utils/file_utils.py
 */
@ApplicationScoped
public class FileTrackingService {

    @Inject
    AppConfig config;

    public String normalizeRelPath(String path) {
        return Paths.get(path).normalize().toString();
    }

    public void updateIngestedFiles(String file) {
        appendToFile(config.getIngestedFile(), normalizeRelPath(file));
    }

    public void updateFailedFiles(String file) {
        appendToFile(config.getFailedFiles(), normalizeRelPath(file));
    }

    public Set<String> loadTracked(String filepath) {
        Path path = Paths.get(filepath);
        if (!Files.exists(path)) {
            return new HashSet<>();
        }
        try {
            return Files.readAllLines(path).stream()
                    .map(String::trim)
                    .filter(line -> !line.isEmpty())
                    .map(this::normalizeRelPath)
                    .collect(Collectors.toSet());
        } catch (IOException e) {
            return new HashSet<>();
        }
    }

    private void appendToFile(String filepath, String content) {
        Path path = Paths.get(filepath);
        try {
            // Ensure parent directory exists
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }

            try (FileChannel channel = FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.APPEND, StandardOpenOption.WRITE)) {
                try (FileLock lock = channel.lock()) {
                    channel.write(java.nio.ByteBuffer.wrap((content + "\n").getBytes(StandardCharsets.UTF_8)));
                }
            }
        } catch (IOException e) {
            // Log error
            System.err.println("Failed to update tracking file " + filepath + ": " + e.getMessage());
        }
    }

    public String md5(String text) {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] hashInBytes = md.digest(text.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder();
            for (byte b : hashInBytes) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("MD5 algorithm not found", e);
        }
    }
}
