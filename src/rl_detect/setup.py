from setuptools import find_packages, setup

package_name = 'rl_detect'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, 'scripts'],
    
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Muhammad Rameez Ur Rahman',
    maintainer_email='rameezrehman83@gmail.com',
    description='Implementation of OpenNav: Efficient Open Vocabulary 3D Object Detection for Smart Wheelchair Navigation paper accepted at ACVR24 ECCV24 workshop',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'intel_publisher_yolo_3dbbox_node2 = scripts.intel_yolo_3dbbox_node2:main',
        ],
    },
)
