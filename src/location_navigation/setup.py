import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'location_navigation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Ament index resource marker
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # package.xml
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        # Config / YAML files
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ali Ahsan Reyan',
    maintainer_email='aliahsanreyam@gmail.com',
    description='Natural-language location navigation for car_bot.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Navigation client standalone
            'navigation_client = location_navigation.navigation_client:main',
            # LLM JSON → Nav2 bridge (primary entry point)
            'llm_to_nav = location_navigation.llm_to_nav:main',
        ],
    },
)
