#
#  FindGLM.cmake
# 
#  Try to find GLM include path.
#  Once done this will define
#
#  GLM_INCLUDE_DIRS
# 
#  Created on 7/17/2014 by Stephen Birarda
#  Copyright 2014 High Fidelity, Inc.
# 
#  Distributed under the Apache License, Version 2.0.
#  See the accompanying file LICENSE or http://www.apache.org/licenses/LICENSE-2.0.html
# 

# setup hints for GLM search
set(GLM_HEADER_SEARCH_HINTS "${GLM_ROOT_DIR}" "$ENV{GLM_ROOT_DIR}" "$ENV{HIFI_LIB_DIR}/glm")

# locate header
find_path(GLM_INCLUDE_DIR "glm/glm.hpp" HINTS ${GLM_HEADER_SEARCH_HINTS})
set(GLM_INCLUDE_DIRS "${GLM_INCLUDE_DIR}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLM DEFAULT_MSG GLM_INCLUDE_DIRS)