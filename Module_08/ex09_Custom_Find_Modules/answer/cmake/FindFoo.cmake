# FindFoo.cmake

# 1. Find Header
find_path(FOO_INCLUDE_DIR NAMES foo.h PATHS ${CMAKE_SOURCE_DIR}/fake_lib/include)

# 2. Find Library
find_library(FOO_LIBRARY NAMES foo PATHS ${CMAKE_SOURCE_DIR}/fake_lib/lib)

# 3. Handle Standard Args (REQUIRED, QUIET, etc.)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Foo
    REQUIRED_VARS FOO_LIBRARY FOO_INCLUDE_DIR
)

# 4. Create Imported Target
if(Foo_FOUND AND NOT TARGET Foo::Foo)
    add_library(Foo::Foo UNKNOWN IMPORTED)
    set_target_properties(Foo::Foo PROPERTIES
        IMPORTED_LOCATION "${FOO_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FOO_INCLUDE_DIR}"
    )
endif()
