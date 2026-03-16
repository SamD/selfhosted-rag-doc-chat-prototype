package com.selfhosted.rag.persistence.service;

import com.selfhosted.rag.common.config.AppConfig;
import com.selfhosted.rag.common.model.ChunkEntry;
import jakarta.annotation.PostConstruct;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.List;

/**
 * Service for DuckDB operations, mirroring Python's parquet_service.py
 */
@ApplicationScoped
public class DuckDbService {

    @Inject
    AppConfig appConfig;

    private String dbUrl;

    @PostConstruct
    void init() {
        dbUrl = "jdbc:duckdb:" + appConfig.getDuckdbFile();
        ensureSchema();
    }

    public void ensureSchema() {
        try (Connection conn = DriverManager.getConnection(dbUrl);
             Statement stmt = conn.createStatement()) {
            
            stmt.execute("""
                CREATE TABLE IF NOT EXISTS parquet_chunks (
                  id VARCHAR PRIMARY KEY,
                  chunk TEXT,
                  source_file VARCHAR,
                  type VARCHAR,
                  chunk_index INTEGER,
                  engine VARCHAR,
                  hash VARCHAR,
                  page INTEGER
                )
                """);
        } catch (SQLException e) {
            System.err.println("❌ Failed to ensure DuckDB schema: " + e.getMessage());
        }
    }

    public synchronized void writeToParquet(List<ChunkEntry> entries, String parquetPath) {
        if (entries == null || entries.isEmpty()) return;

        try (Connection conn = DriverManager.getConnection(dbUrl)) {
            // 1. Insert/Upsert into DuckDB
            String upsertSql = """
                INSERT INTO parquet_chunks (id, chunk, source_file, type, chunk_index, engine, hash, page)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    chunk = excluded.chunk,
                    source_file = excluded.source_file,
                    type = excluded.type,
                    chunk_index = excluded.chunk_index,
                    engine = excluded.engine,
                    hash = excluded.hash,
                    page = excluded.page
                """;
            
            try (PreparedStatement pstmt = conn.prepareStatement(upsertSql)) {
                for (ChunkEntry entry : entries) {
                    pstmt.setString(1, entry.getId());
                    pstmt.setString(2, entry.getChunk());
                    pstmt.setString(3, entry.getSource_file());
                    pstmt.setString(4, entry.getType());
                    pstmt.setInt(5, entry.getChunk_index() != null ? entry.getChunk_index() : -1);
                    pstmt.setString(6, entry.getEngine());
                    pstmt.setString(7, entry.getHash());
                    pstmt.setInt(8, entry.getPage() != null ? entry.getPage() : -1);
                    pstmt.addBatch();
                }
                pstmt.executeBatch();
            }

            // 2. Export the entire table to Parquet (overwriting the file)
            // Mirroring Python's COPY behavior
            try (Statement stmt = conn.createStatement()) {
                stmt.execute(String.format("COPY parquet_chunks TO '%s' (FORMAT PARQUET)", parquetPath));
            }

            System.out.println("✅ Archived " + entries.size() + " chunks to DuckDB and " + parquetPath);

        } catch (SQLException e) {
            System.err.println("❌ Failed to write to DuckDB/Parquet: " + e.getMessage());
        }
    }
}
