# Packages required by NiFi's Java-to-Python bridge
from nifiapi.properties import PropertyDescriptor
from nifiapi.processor import ProcessorDetails, FlowFileTransform, FlowFileTransformResult

class PythonHttpRequest(FlowFileTransform):
    class Details(ProcessorDetails):
        name = "PythonHttpRequest"
        version = "2.0.0"
        description = "Executes an HTTP request using the native Python requests library."
        tags = ["http", "request", "python", "client"]
        dependencies = ['requests==2.32.3'] # NiFi automatically pip installs this dependency

    # Define user-configurable UI parameters
    HTTP_URL = PropertyDescriptor(
        name="HTTP URL",
        description="The remote URL to send the HTTP request to.",
        required=True
    )
    HTTP_METHOD = PropertyDescriptor(
        name="HTTP Method",
        description="The HTTP method to use.",
        allowable_values=["GET", "POST", "PUT", "DELETE"],
        default_value="GET",
        required=True
    )

    def __init__(self, **kwargs):
        super().__init__()
        self.descriptors = [self.HTTP_URL, self.HTTP_METHOD]

    def getPropertyDescriptors(self):
        return self.descriptors

    def transform(self, context, flowfile):
        import requests  # Import inside the execution method

        # Retrieve configured parameters from the NiFi context
        url = context.getProperty(self.HTTP_URL).getValue()
        method = context.getProperty(self.HTTP_METHOD).getValue()

        # Read the incoming FlowFile payload to use as the request body
        body_bytes = flowfile.read()

        # Extract existing attributes to pass along or inject as headers
        attributes = flowfile.getAttributes()

        try:
            # Execute the network call
            if method == "GET":
                response = requests.get(url, timeout=30)
            elif method == "POST":
                response = requests.post(url, data=body_bytes, timeout=30)
            elif method == "PUT":
                response = requests.put(url, data=body_bytes, timeout=30)
            elif method == "DELETE":
                response = requests.delete(url, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Capture API response values
            response_text = response.text
            response_status = str(response.status_code)

            # Update FlowFile metadata attributes
            attributes["http.status.code"] = response_status

            if response.status_code >= 400:
                # Route failed HTTP statuses to the failure relationship
                return FlowFileTransformResult(
                    relationship="failure",
                    contents=response_text.encode('utf-8'),
                    attributes=attributes
                )

            # Route successful network calls to the success relationship
            return FlowFileTransformResult(
                relationship="success",
                contents=response_text.encode('utf-8'),
                attributes=attributes
            )

        except Exception as e:
            # Catch network layer dropouts or DNS errors
            attributes["http.error"] = str(e)
            return FlowFileTransformResult(
                relationship="failure",
                contents=f"Execution error: {str(e)}".encode('utf-8'),
                attributes=attributes
            )
