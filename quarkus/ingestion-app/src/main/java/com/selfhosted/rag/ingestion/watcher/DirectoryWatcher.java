package com.selfhosted.rag.ingestion.watcher;

import com.selfhosted.rag.common.config.AppConfig;
import com.selfhosted.rag.common.service.FileTrackingService;
import com.selfhosted.rag.ingestion.service.ChunkQueueService;
import io.quarkus.scheduler.Scheduled;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Stream;

/**
 * Directory watcher that scans for new files to ingest.
 *
 * Maps to Python: workers/producer_worker.py - run_tree_watcher()
 */
@ApplicationScoped
public class DirectoryWatcher {

    @Inject
    AppConfig appConfig;

    @Inject
    FileTrackingService fileTrackingService;

    @Inject
    ChunkQueueService chunkQueueService;

    @Inject
    org.eclipse.microprofile.context.ManagedExecutor managedExecutor;

    private final AtomicBoolean isScanning = new AtomicBoolean(false);

    @Scheduled(every = "30s")
    public void scanDirectory() {
        if (!appConfig.isProducerEnabled()) {
            return;
        }
        
        if (isScanning.compareAndSet(false, true)) {
            try {
                runTreeWatcher();
            } finally {
                isScanning.set(false);
            }
        }
    }

    private void runTreeWatcher() {
        String ingestFolder = appConfig.getIngestFolder();
        if (ingestFolder == null || ingestFolder.isBlank()) {
            System.err.println("Ingest folder not configured");
            return;
        }

        Path ingestPath = Paths.get(ingestFolder);
        if (!Files.exists(ingestPath)) {
            System.err.println("Ingest folder does not exist: " + ingestFolder);
            return;
        }

        Set<String> tracked = fileTrackingService.loadTracked(appConfig.getTrackFile());
        Set<String> failed = fileTrackingService.loadTracked(appConfig.getFailedFiles());

        try (Stream<Path> paths = Files.walk(ingestPath)) {
            paths.filter(Files::isRegularFile)
                    .filter(path -> isSupported(path.toString()))
                    .forEach(path -> {
                        String relPath = fileTrackingService.normalizeRelPath(ingestPath.relativize(path).toString());
                        if (!tracked.contains(relPath) && !failed.contains(relPath)) {
                            // Submit to executor for parallel processing, matching Python Pool behavior
                            managedExecutor.execute(() -> {
                                chunkQueueService.ingestFile(path.toString(), relPath, UUID.randomUUID().toString());
                            });
                        }
                    });
        } catch (IOException e) {
            System.err.println("Error walking ingest folder: " + e.getMessage());
        }
    }

    private boolean isSupported(String filename) {
        String lower = filename.toLowerCase();
        return appConfig.getSupportedExtensionsList().stream().anyMatch(lower::endsWith);
    }
}
