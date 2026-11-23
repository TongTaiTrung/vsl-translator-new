class StreamProcess:
    def __init__(self):
        self.latest_result = None
        self.latest_timestamp = None
        self.output_img = None
    def update(self, results, output_img, timestamp_ms):
        self.latest_result,self.output_img,self.latest_timestamp = (
            results,
            output_img,
            timestamp_ms
        )
    def get(self):
        return self.latest_result