# AGENTS.md

## Build, Lint, and Test Commands

### Maven Build Commands
```bash
# Build entire project
mvn clean install -DskipTests

# Build specific module
cd quarkus/front-end && mvn clean install -DskipTests
cd quarkus/ingestion-app && mvn clean install -DskipTests
cd quarkus/persistence-app && mvn clean install -DskipTests
cd quarkus/ocr-fallback-app && mvn clean install -DskipTests
cd quarkus/common-lib && mvn clean install -DskipTests
```

### Lint Commands
```bash
# Java linting (checkstyle, spotbugs, etc.)
mvn checkstyle:check
mvn spotbugs:check

# Python linting
ruff check .
```

### Test Commands
```bash
# Run all tests
mvn test

# Run tests in specific module
cd quarkus/front-end && mvn test
cd quarkus/ingestion-app && mvn test
cd quarkus/persistence-app && mvn test

# Run single test class
mvn test -Dtest=ChatResourceTest

# Run single test method
mvn test -Dtest=ChatResourceTest#testHealthEndpoint

# Run tests with coverage
mvn clean test jacoco:report
```

## Code Style Guidelines

### Java (Quarkus Backend)

**Imports**
- Group imports: standard library → third-party → project-specific
- Use wildcard imports for static methods only
- Sort imports alphabetically within groups
- Place `package` statement first, then imports

**Formatting**
- Line length: 270 characters (from ruff.toml)
- Indentation: 4 spaces
- Braces: K&R style (opening brace on same line as method/class declaration)
- Use Lombok annotations for boilerplate reduction

**Naming Conventions**
- Classes: PascalCase (e.g., `ChatResource`, `QueryRequest`)
- Methods: camelCase (e.g., `answerQuery`, `getQuery`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_CHUNK_SIZE`)
- Package names: lowercase (e.g., `com.selfhosted.rag.api.resource`)

**Type Annotations**
```java
@Data                    // Generate getters/setters
@Builder                   // Builder pattern
@NoArgsConstructor            // No-args constructor
@AllArgsConstructor            // All-args constructor
```

**Error Handling**
- Use Jakarta REST exceptions with proper HTTP status codes
- Log errors with `System.out.println` or proper logging framework
- Return structured error responses in JSON format

**Code Structure**
```java
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
@Path("/api/v1")
public class ChatResource {
    @Inject
    RagService ragService;
    
    @GET
    @Path("/health")
    public Map<String, String> health() { ... }
    
    @POST
    @Path("/query")
    public QueryResponse query(QueryRequest req) { ... }
}
```

### Frontend (Astro)

**Formatting**
- Use Tailwind CSS utility classes
- Follow Astro component conventions
- TypeScript for type safety

### Python (if applicable)
- Follow ruff.toml configuration (line-length: 270)
- Use type hints where appropriate

## Project Structure

```
selfhosted-rag-doc-chat-prototype/
├── quarkus/                    # Java backend
│   ├── common-lib/             # Shared models and utilities
│   ├── front-end/              # REST API
│   ├── ingestion-app/       # Document ingestion
│   ├── persistence-app/        # Vector storage
│   └── ocr-fallback-app/      # OCR processing
├── astro-frontend/            # Web UI
└── doc-ingest-chat/           # Python reference implementation
```

## Testing Approach

- Use JUnit 5 for unit tests
- Mock dependencies with `@InjectMock`
- REST API tests with RestAssured
- AssertJ for assertions
- Test both happy paths and error conditions