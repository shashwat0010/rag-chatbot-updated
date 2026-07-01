from locust import HttpUser, task, between

class FastApiUser(HttpUser):
    # Simulate a user waiting 1 to 3 seconds between requests
    wait_time = between(1, 3)

    @task
    def health_check(self):
        # We test the health endpoint to check base application performance
        # without consuming API credits for LLM calls.
        self.client.get("/health")
