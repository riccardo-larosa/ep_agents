import yaml
import requests

class YamlLoader:
    def __init__(self, urls: list[str]):
        self.urls = urls
        self._cached_data = None  # Cache

    def load_yaml_files(self):
        yaml_data = {}
        for url in self.urls:
            try:
                response = requests.get(url)
                response.raise_for_status()  # Raise an error for bad status codes
                # Load the YAML content and store it in the dictionary using the URL as the key
                yaml_data[url] = yaml.safe_load(response.text)
            except requests.exceptions.RequestException as e:
                print(f"Failed to load {url}: {e}")
        return yaml_data

    def get_data(self):
        if self._cached_data is None:
            self._cached_data = self.load_yaml_files()  # Cache the data
        return self._cached_data