function(add_sanitizer_flags)
    if(NOT ${ENABLE_SANITIZE_ADDRESS} AND NOT ${ENABLE_SANITIZE_UNDEFINED})
        message(STATUS "Sanitizers disabled for: ${TARGET}")
        return()
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "CLANG" OR CMAKE_CXX_COMPILER_ID MATCHES
                                                "GNU")
        messages(STATUS "Sanitizers activated")
        add_compile_options("-fno-omit-frame-pointer")
        add_link_options("-fno-omit-frame-pointer")

        if(${ENABLE_SANITIZE_ADDRESS})
            add_compile_options("-fsanitize=address")
            add_link_options("-fsanitize=address")
        endif()

        if(${ENABLE_SANITIZE_UNDEFINED})
            add_compile_options("-fsanitize=undefined")
            add_link_options("-fsanitize=undefined")
        endif()

        if(ENABLE_SANITIZE_LEAK)
            add_compile_options("-fsanitize=leak")
            add_link_options("-fsanitize=leak")
        endif()

        if(ENABLE_SANITIZE_THREAD)
            if(ENABLE_SANITIZE_ADDR OR ENABLE_SANITIZE_LEAK)
                message(WARNING "thread does not work with: address and leak")
            endif()
            add_compile_options("-fsanitize=thread")
            add_link_options("-fsanitize=thread")
        endif()

    else()
        messages(ERROR "Compiler not supported for Sanitizers")
    endif()

    target_compile_options(${TARGET} PRIVATE ${WARNINGS})
    message(STATUS ${WARNINGS})

endfunction()
