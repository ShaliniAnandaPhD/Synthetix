from setuptools import setup, find_packages

setup(
    name="neuron-core",
    version="1.0.4",
    # Map the neuron_core package to the current directory
    packages=[
        'neuron_core',
        'neuron_core.agents',
        'neuron_core.core',
        'neuron_core.memory',
    ],
    package_dir={
        'neuron_core': '.',
        'neuron_core.agents': 'agents',
        'neuron_core.core': 'core',
        'neuron_core.memory': 'memory',
    },
    install_requires=[
        "google-cloud-aiplatform",
        "google-cloud-firestore>=2.11.0",
        "pydantic",
        "numpy",
        "opentelemetry-api>=1.20.0",
        "opentelemetry-sdk>=1.20.0",
        "opentelemetry-exporter-gcp-trace>=1.5.0",
        "opentelemetry-instrumentation>=0.41b0"
    ],
    author="ShaliniAnandaPhD",
    description="Core cognitive architecture for Neuron framework",
)
