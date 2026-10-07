import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'car_bot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        ('share/' + package_name + '/urdf', glob('urdf/*.xacro')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config', glob('config/*.rviz')),
        ('share/' + package_name + '/world', glob('world/*.sdf')),
        ('share/' + package_name + '/map', glob('map/*')),


    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='safin',
    maintainer_email='safin@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
             'twist_to_stamped = car_bot.twist_to_stamped:main',
        ],
    },
)
