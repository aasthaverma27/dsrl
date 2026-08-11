FROM nvgear/gr00t_wbc:latest

# Isaac-GR00T needs transformers 4.57.x; the base image ships none.
# 5.x breaks Gr00tN1d7Config (dataclass over PretrainedConfig).
RUN /root/venv/bin/pip install --no-cache-dir "transformers==4.57.3"

# ROS lives in /opt/ros/humble but the base image only sources it from
# .bashrc, so non-interactive runs fail with "No module named rclpy".
# ROS_LOCALHOST_ONLY must match the WBC process or DDS participants
# cannot discover each other.
ENV ROS_DOMAIN_ID=99 \
    ROS_LOCALHOST_ONLY=1 \
    PYTHONPATH=/opt/isaac-gr00t

RUN printf '%s\n' \
    '#!/bin/bash' \
    'source /opt/ros/humble/setup.bash' \
    'export PYTHONPATH=/opt/isaac-gr00t:$PYTHONPATH' \
    'exec "$@"' > /usr/local/bin/entry.sh && chmod +x /usr/local/bin/entry.sh

ENTRYPOINT ["/usr/local/bin/entry.sh"]
CMD ["bash"]
