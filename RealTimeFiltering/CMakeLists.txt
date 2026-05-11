# Dodajemy main.cpp do listy plików
add_executable (RealTimeFiltering 
    "main.cpp"
    "RealTimeFiltering.cpp" 
    "RealTimeFiltering.h"
    "CudaFilters.cu"    
    "CudaFilters.h"     
)

# Łączymy z bibliotekami
target_link_libraries(RealTimeFiltering PRIVATE 
    Qt6::Widgets 
    ${OpenCV_LIBS}
)

# Żeby Visual Studio widziało plik wideo w folderze z projektem
set_target_properties(RealTimeFiltering PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# 1. Qt - Używamy narzędzia windeployqt
# To narzędzie kopiuje wszystko: DLL-ki, foldery 'platforms', 'styles' itp.
if(WIN32)
    # Szukamy gdzie jest qmake, żeby znaleźć folder bin od Qt
    get_target_property(_qmake_executable Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(_qt_bin_dir "${_qmake_executable}" DIRECTORY)
    
    # Wybieramy flagę --debug lub --release w zależności od konfiguracji VS
    set(DEPLOY_FLAGS "")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(DEPLOY_FLAGS "--debug")
    else()
        set(DEPLOY_FLAGS "--release")
    endif()

    add_custom_command(TARGET RealTimeFiltering POST_BUILD
        COMMAND "${_qt_bin_dir}/windeployqt.exe" ${DEPLOY_FLAGS} --no-translations --no-compiler-runtime "$<TARGET_FILE:RealTimeFiltering>"
        COMMENT "Automatyczne wdrażanie bibliotek Qt..."
    )
endif()

# 2. OpenCV - Kopiowanie konkretnej DLL-ki
if(WIN32)
    # Ponieważ OpenCV_DIR wskazuje na folder 'lib', DLL-ka jest w folderze '../bin'
    # UWAGA: Sprawdź numer swojej wersji OpenCV! Jeśli masz 4.12.0, to będzie 4120.
    set(OPENCV_DLL_NAME "opencv_world4120d.dll") 
    
    # Budujemy ścieżkę do binariów OpenCV
    get_filename_component(OPENCV_BIN_DIR "${OpenCV_DIR}/../bin" REALPATH)

    add_custom_command(TARGET RealTimeFiltering POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPENCV_BIN_DIR}/${OPENCV_DLL_NAME}"
        "$<TARGET_FILE_DIR:RealTimeFiltering>"
        COMMENT "Kopiowanie OpenCV DLL do folderu exe..."
    )
endif()