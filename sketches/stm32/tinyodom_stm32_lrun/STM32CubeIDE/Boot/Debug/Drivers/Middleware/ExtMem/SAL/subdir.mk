################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../../../Middlewares/ST/STM32_ExtMem_Manager/sal/stm32_sal_xspi.c 

OBJS += \
./Drivers/Middleware/ExtMem/SAL/stm32_sal_xspi.o 

C_DEPS += \
./Drivers/Middleware/ExtMem/SAL/stm32_sal_xspi.d 


# Each subdirectory must supply rules for building sources it contributes
Drivers/Middleware/ExtMem/SAL/stm32_sal_xspi.o: ../../../Middlewares/ST/STM32_ExtMem_Manager/sal/stm32_sal_xspi.c Drivers/Middleware/ExtMem/SAL/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m55 -std=gnu11 -g3 -DNO_OTP_FUSE -DDEBUG -DSTM32N657xx -DSTM32N6 -DSTM32 -DSTM32N6xx -c -I../Inc -I../../../FSBL/Inc -I../../../Drivers/CMSIS/Include -I../../../Drivers/CMSIS/Device/ST/STM32N6xx/Include -I../../../Drivers/STM32N6xx_HAL_Driver/Inc -I../../../Middlewares/ST/STM32_ExtMem_Manager -I../../../Middlewares/ST/STM32_ExtMem_Manager/SAL -I../../../Middlewares/ST/STM32_ExtMem_Manager/NOR_SFDP -I../../../Middlewares/ST/STM32_ExtMem_Manager/boot -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -mcmse -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Drivers-2f-Middleware-2f-ExtMem-2f-SAL

clean-Drivers-2f-Middleware-2f-ExtMem-2f-SAL:
	-$(RM) ./Drivers/Middleware/ExtMem/SAL/stm32_sal_xspi.cyclo ./Drivers/Middleware/ExtMem/SAL/stm32_sal_xspi.d ./Drivers/Middleware/ExtMem/SAL/stm32_sal_xspi.o ./Drivers/Middleware/ExtMem/SAL/stm32_sal_xspi.su

.PHONY: clean-Drivers-2f-Middleware-2f-ExtMem-2f-SAL

